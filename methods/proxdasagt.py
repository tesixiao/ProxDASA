"""
Decentralized Proximal Averaged Stochastic Approximation with Gradient Tracking

"""

# Import packages
from __future__ import print_function
import argparse
import os
import sys
import time
import torch
import numpy
import math
from mpi4py import MPI
from torchvision import datasets, transforms

# Import custom classes
from models.mlp import MLP
from models.lenet import LENET
from helpers.l1_regularizer import L1
from helpers.replace_weights import Opt
from helpers.custom_data_loader import BinaryDataset

# Set up MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Declare new method

class PROXDASAGT:
    '''
    Class for PROXDASAGT

    :param: local_params = DICT of parameters for training
    :param: mixing_matrix = NxN torch float containing weights for communication
    :param: training_data = torch.utils.data.Dataloader
    :param: init_weights = LIST of NUMPY arrays containing initial weights for the network
    '''

    def __init__(self, local_params, mixing_matrix, training_data, init_weights):

        # Get the information about neighbor communication:
        # First, we extract the number of nodes and double check
        # this value is the same as the size of the MPI world
        # Second, we extract thr row of the mixing matrix corresponding to this agent
        # and save the weights
        self.mixing_matrix = mixing_matrix.float()
        self.num_nodes = self.mixing_matrix.shape[0]
        if self.num_nodes != size:
            sys.exit(f"Cannot match MPI size {size} with mixing matrix of shape {self.num_nodes}. ")
        self.peers = torch.where(self.mixing_matrix[rank, :] != 0)[0].tolist()
        self.peers.remove(rank)
        self.peer_weights = self.mixing_matrix[rank, self.peers].tolist()
        self.my_weight = self.mixing_matrix[rank, rank].item()

        ##################################################
        # Parse the training parameters:
        #
        # model = 'bilinear', 'lenet', or 'fc' for architecture type (STR)
        # step_type = 'constant' or 'diminishing' for learning rate type (STR)
        # alpha = learning rate (FLOAT)
        # mini_batch = batch size (INT)
        # l1 = regularization coefficient (FLOAT)
        # report = how often to report stationarity, test acc, etc. (INT)
        ##################################################
        if 'step_type' in local_params:
            self.step_type = local_params['step_type']
        else:
            self.step_type = 'constant'
        if 'mini_batch' in local_params:
            self.mini_batch = int(local_params['mini_batch'])
        else:
            self.mini_batch = 128
        if 'l1' in local_params:
            self.l1 = local_params['l1']
        else:
            self.l1 = 0.0
        if 'report' in local_params:
            self.report = local_params['report']
        else:
            self.report = 100
        if 'comm_round' in local_params:
            self.comm_round = local_params['comm_round']
        else:
            rho = torch.sort(torch.linalg.eig(mixing_matrix)[0].real)[0][size - 2].item()
            self.comm_round = int(math.log(2 * self.num_nodes + 3) / (2 - 2*rho))
        self.lr = local_params['lr']

        # Get the CUDA device and save the data loader to be easily reference later
        # self.device = torch.device(f'cuda:{rank % size}')
        
        # CPU
        self.device = torch.device(f'cpu:{rank % size}')

        self.alpha_base = local_params['alpha_base']
        self.data_loader = training_data

        # Initialize the models
        # We either have the MLP or we have LENET

        if args.data == 'a9a':
            self.model = MLP(self.data_loader.dataset.data.shape[1], 64, 2).to(self.device)

        elif args.data == 'mnist':
            self.model = LENET(10).to(self.device)

        else:
            sys.exit(f"[ERROR] To use a new dataset/architecture, add the dataset to the data folder and incorporate the"
                     f"model here using \'self.model = <your_model>.to(self.device)\'.")

        # Initialize the updating weights rule and the training loss function
        self.replace_weights = Opt(self.model.parameters(), lr=0.1)
        self.training_loss_function = torch.nn.NLLLoss(reduction='mean')

        # Initialize the l1 regularizer
        self.regularizer = L1(self.device)

        # Load variables
        self.weights = [torch.tensor(init_weights[i]).to(self.device) for i in range(len(init_weights))]

        # Save number of parameters
        self.num_params = len(self.weights)

        self.grads = self.get_grads(self.weights)
        
        # Initialize local y, z, u, and previous grads
        self.Y = [self.weights[k].detach() for k in range(self.num_params)]
        # self.Z = [self.weights[k].detach() for k in range(self.num_params)]
        self.Z = [torch.zeros_like(self.weights[k]) for k in range(self.num_params)]
        self.U = [self.grads[k].detach() for k in range(self.num_params)]
        self.prev_grads = [self.grads[k].detach() for k in range(self.num_params)]

        # Allocate space for relevant report values: consensus, gradient,
        # iterate norm, number non-zeros, training/testing acc, compute time, etc.
        self.consensus_violation = []
        self.norm_hist = []
        self.total_optimality = []
        self.iterate_norm_hist = []
        self.nnz_at_avg = []
        self.avg_nnz = []
        self.testing_loss = []
        self.testing_accuracy = []
        self.training_loss = []
        self.training_accuracy = []
        self.testing_loss_local = []
        self.testing_accuracy_local = []
        self.training_loss_local = []
        self.training_accuracy_local = []
        self.compute_time = []
        self.communication_time = []
        self.total_time = []

    def solve(self, outer_iterations, training_data_full_sample, testing_data):
        '''Solve the global problem'''

        ##################################################
        # Save initial errors for fair comparison across methods
        avg_weights = self.get_average_param(self.weights)
        cons, norm, total, var_norm, nnz_at_avg, avg_nnz = self.compute_optimality_criteria(avg_weights, self.weights,
                                                                                            training_data_full_sample)
        self.consensus_violation.append(cons)
        self.norm_hist.append(norm)
        self.total_optimality.append(total)
        self.iterate_norm_hist.append(var_norm)
        self.nnz_at_avg.append(nnz_at_avg)
        self.avg_nnz.append(avg_nnz)

        # TEST ACCURACY ON TRAINING SET
        train_loss, train_acc = self.test(avg_weights, self.data_loader)
        self.training_loss.append(train_loss)
        self.training_accuracy.append(train_acc)

        # TEST ACCURACY ON TEST SET
        test_loss, test_acc = self.test(avg_weights, testing_data)
        self.testing_loss.append(test_loss)
        self.testing_accuracy.append(test_acc)

        # TEST ACCURACY ON TRAINING SET AT LOCAL
        train_loss_local, train_acc_local = self.test(self.weights, self.data_loader, mode='local')
        self.training_loss_local.append(train_loss_local)
        self.training_accuracy_local.append(train_acc_local)

        # TEST ACCURACY ON TEST SET AT LOCAL
        test_loss_local, test_acc_local = self.test(self.weights, testing_data, mode='local')
        self.testing_loss_local.append(test_loss_local)
        self.testing_accuracy_local.append(test_acc_local)
        ##################################################

        if rank == 0:
        # First iteration, print headings, then print the values
            print("{:<10} | {:<7} | {:<13} | {:<15} | {:<15} | {:<15} | {:<15} | {:<12} | {:<6}".format("Iteration", "Epoch",
                                                                                      "Stationarity",
                                                                                      "Train (L / A)",
                                                                                      "Test (L / A)",
                                                                                      "Train (L / A) L",
                                                                                      "Test (L / A) L",
                                                                                      "Avg Density",
                                                                                      "Time"))
            print("{:<10} | {:<7} | {:<13} | {:<15} | {:<15} | {:<15} | {:<15} | {:<12} | {:<6}".format(0,
                            round(0, 2),
                            round(self.total_optimality[0], 4),
                            f"{round(self.training_loss[0], 4)} / {round(self.training_accuracy[0], 2)}",
                            f"{round(self.testing_loss[0], 4)} / {round(self.testing_accuracy[0], 2)}",
                            f"{round(self.training_loss[0], 4)} / {round(self.training_accuracy[0], 2)}",
                            f"{round(self.testing_loss[0], 4)} / {round(self.testing_accuracy_local[0], 2)}",
                            round(self.avg_nnz[0], 6),
                            0.0))


        # Time the entire algorithm
        t0 = time.time()

        # Barrier communication at beginning of run to force agents to start at the same time
        comm.Barrier()

        # alpha
        self.alpha = self.alpha_base * math.sqrt(self.num_nodes / outer_iterations)

        # Loop over algorithm updates
        for i in range(outer_iterations):
            # Local updates
            if self.step_type == 'constant':
                pass
            else:
                self.alpha = self.alpha_base * min(math.sqrt(self.num_nodes / (i+1)), 1)
            # Time part of computation time
            time_i = time.time()

            ##################################################
            # Update local y via prox
            self.Y = self.regularizer.forward(
                [self.weights[k].detach().clone() - self.lr * self.Z[k].detach().clone() for k in
                 range(self.num_params)], self.lr * self.l1)

            # Obtain local grads
            self.grads = self.get_grads(self.weights)

            # Update local x via moving average
            self.weights = [(1 - self.alpha) * self.weights[k].detach().clone() + \
                                    self.alpha * self.Y[k].detach().clone() for k in range(self.num_params)]

            # Update local z via moving average
            self.Z = [(1 - self.alpha) * self.Z[k].detach().clone() + \
                                    self.alpha * self.grads[k] for k in range(self.num_params)]

            # Update u via gradient tracking
            self.U = [self.U[k] + self.grads[k] - self.prev_grads[k] for k in range(self.num_params)]

            # Save pre grads for gradient tracking
            self.prev_grads = [self.grads[pa].detach().clone() for pa in range(len(self.grads))]
            ##################################################


            # STOP TIME FOR COMPUTING
            time_i_end = time.time()
            comm_time1 = 0
            ##################################################
            # Communication
            for _ in range(self.comm_round):
                # Communicate X, U, and Z
                comm.Barrier()
                comm_time1 += self.communicate_with_neighbors()
                comm.Barrier()
            ##################################################

            # Save pre communication weights for computing nnz
            pre_comm_weights = [self.weights[k].detach().clone() for k in range(self.num_params)]

            # Save times
            comp_time = round(time_i_end - time_i, 4)
            comm_time = comm_time1
            ##################################################

            # Barrier at the end of update for extreme safety
            comm.Barrier()

            # Save values at report interval
            if i % self.report == 0 and i > 0:

                # Save the first errors using the average value - so all agents are compared fairly
                avg_weights = self.get_average_param(self.weights)
                cons, norm, total, var_norm, nnz_at_avg, avg_nnz = self.compute_optimality_criteria(avg_weights,
                                                                                                    self.weights,
                                                                                                    training_data_full_sample,
                                                                                                    pre_comm_weights)
                self.consensus_violation.append(cons)
                self.norm_hist.append(norm)
                self.total_optimality.append(total)
                self.iterate_norm_hist.append(var_norm)
                self.nnz_at_avg.append(nnz_at_avg)
                self.avg_nnz.append(avg_nnz)

                # TEST ACCURACY ON TRAINING SET
                train_loss, train_acc = self.test(avg_weights, self.data_loader)
                self.training_loss.append(train_loss)
                self.training_accuracy.append(train_acc)

                # TEST ACCURACY ON TEST SET
                test_loss, test_acc = self.test(avg_weights, testing_data)
                self.testing_loss.append(test_loss)
                self.testing_accuracy.append(test_acc)

                # TEST ACCURACY ON TRAINING SET AT LOCAL
                train_loss_local, train_acc_local = self.test(self.weights, self.data_loader, mode='local')
                self.training_loss_local.append(train_loss_local)
                self.training_accuracy_local.append(train_acc_local)

                # TEST ACCURACY ON TEST SET AT LOCAL
                test_loss_local, test_acc_local = self.test(self.weights, testing_data, mode='local')
                self.testing_loss_local.append(test_loss_local)
                self.testing_accuracy_local.append(test_acc_local)

                # Print relevant information
                if rank == 0:
                    print("{:<10} | {:<7} | {:<13} | {:<15} | {:<15} | {:<15} | {:<15} | {:<12} | {:<6}".format(i,
                                    round((i * self.mini_batch) / (self.data_loader.dataset.data.shape[0] // size), 2),
                                    round(total, 4),
                                    f"{round(train_loss, 4)} / {round(train_acc, 2)}",
                                    f"{round(test_loss, 4)} / {round(test_acc, 2)}",
                                    f"{round(train_loss_local, 4)} / {round(train_acc_local, 2)}",
                                    f"{round(test_loss_local, 4)} / {round(test_acc_local, 2)}",
                                    round(avg_nnz, 6),
                                    round(time.time() - t0, 1)))

            # Append timing information for each iteration
            sys.stdout.flush()
            self.compute_time.append(comp_time)
            self.communication_time.append(comm_time)
            self.total_time.append(comp_time + comm_time)

        ##################################################
        # End total training time
        t1 = time.time() - t0
        if rank == 0:
            closing_statement = f' Training finished '
            print('\n' + closing_statement.center(50, '-'))
            print(f'[TOTAL TIME] {round(t1, 2)}')

        # Return the training time
        return t1

    def communicate_with_neighbors(self):

        # Time this communication
        time0 = MPI.Wtime()

        # Loop over all of the variables
        for pa in range(self.num_params):

            # DEFINE VARIABLE TO SEND
            send_data = self.weights[pa].cpu().detach().numpy()
            recv_data = numpy.empty(shape=((len(self.peers),) + self.weights[pa].shape), dtype=numpy.float32)
            send_data_z = self.Z[pa].cpu().detach().numpy()
            recv_data_z = numpy.empty(shape=((len(self.peers),) + self.Z[pa].shape), dtype=numpy.float32)
            send_data_u = self.U[pa].cpu().detach().numpy()
            recv_data_u = numpy.empty(shape=((len(self.peers),) + self.U[pa].shape), dtype=numpy.float32)

            # SET UP REQUESTS TO INSURE CORRECT SENDS/RECVS
            recv_request = [MPI.REQUEST_NULL for _ in range(int(2 * len(self.peers)))]
            recv_request_z = [MPI.REQUEST_NULL for _ in range(int(2 * len(self.peers)))]
            recv_request_u = [MPI.REQUEST_NULL for _ in range(int(2 * len(self.peers)))]


            # SEND THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Send the data
                recv_request[ind + len(self.peers)] = comm.Isend(send_data, dest=peer_id)
                recv_request_z[ind + len(self.peers)] = comm.Isend(send_data_z, dest=peer_id)
                recv_request_u[ind + len(self.peers)] = comm.Isend(send_data_u, dest=peer_id)


            # RECEIVE THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Receive the data
                recv_request[ind] = comm.Irecv(recv_data[ind, :], source=peer_id)
                recv_request_z[ind] = comm.Irecv(recv_data_z[ind, :], source=peer_id)
                recv_request_u[ind] = comm.Irecv(recv_data_u[ind, :], source=peer_id)


            # HOLD UNTIL ALL COMMUNICATIONS COMPLETE
            MPI.Request.waitall(recv_request)
            MPI.Request.waitall(recv_request_z)
            MPI.Request.waitall(recv_request_u)

            # SCALE CURRENT WEIGHTS
            self.weights[pa] = self.my_weight * self.weights[pa]
            self.Z[pa] = self.my_weight * self.Z[pa]
            self.U[pa] = self.my_weight * self.U[pa]

            # Update global variables
            for ind in range(len(self.peers)):
                self.weights[pa] += (self.peer_weights[ind] * torch.tensor(recv_data[ind, :]).to(self.device))
                self.Z[pa] += (self.peer_weights[ind] * torch.tensor(recv_data_z[ind, :]).to(self.device))
                self.U[pa] += (self.peer_weights[ind] * torch.tensor(recv_data_u[ind, :]).to(self.device))

        return round(MPI.Wtime() - time0, 4)

    def get_average_param(self, list_of_params):
        '''Perform ALLREDUCE of neighbor parameters'''

        # Save information to blank list
        output_list_of_parameters = [None] * len(list_of_params)

        # Loop over the parameters
        for pa in range(self.num_params):

            # Prep send and receive to be numpy arrays
            send_data = list_of_params[pa].cpu().detach().numpy()
            recv_data = numpy.empty(shape=(list_of_params[pa].shape), dtype=numpy.float32)

            # Barriers and note that the allreduce operations is summation!
            comm.Barrier()
            comm.Allreduce(send_data, recv_data)
            comm.Barrier()

            # Save information by dividing by number of agents and converting to tensor
            output_list_of_parameters[pa] = (1 / self.num_nodes) * torch.tensor(recv_data).to(self.device)

        return output_list_of_parameters

    def get_grads(self, current_weights):
        '''Get a local gradient'''

        # Set model to training mode
        self.model.train()

        # Choose one random sample
        for batch_idx, (data, target) in enumerate(self.data_loader):

            # Print errors
            torch.autograd.set_detect_anomaly(True)

            # Convert data to CUDA if possible
            data, target = data.to(self.device).float(), target.to(self.device).long()

            # Zero out gradients
            self.replace_weights.step(current_weights, self.device)
            self.replace_weights.zero_grad()
            # Forward pass of the model
            out1 = self.model(data)
            loss1 = (1 / self.num_nodes) * self.training_loss_function(out1, target)
            # Compute the gradients
            loss1.backward()

            # Update D
            grads = [p.grad.data.detach().clone() for ind, p in enumerate(self.model.parameters())]
            break

        return grads

    def compute_optimality_criteria(self, avg_weights, local_weights, training_data_full_sample, pre_comm_weights=None):
        '''
        Compute the relevant metrics for this problem

        :param avg_weights: LIST of average weights
        :param local_weights: LIST of local weights
        :param training_data_full_sample: data loader with full gradient size
        :return:
        '''

        # Compute consensus for this agent
        local_violation = sum([numpy.linalg.norm(
            local_weights[i].cpu().numpy().flatten() - avg_weights[i].cpu().numpy().flatten(), ord=2) ** 2 for i in
                               range(len(local_weights))])

        # Compute the norm of the iterate to save in case consensus is large
        avg_weight_norm = sum([numpy.linalg.norm(avg_weights[i].cpu().numpy().flatten(), ord=2) ** 2 for i in
                               range(len(avg_weights))])

        # Compute the gradient at the average solution on this dataset:
        # 1. Replace the model params
        # 2. Forward pass, backward pass to have gradient
        # 3. Compute the stationarity violation
        # 4. MUST SCALE: total number of samples is (N * num_local) samples. Since `get_average_param` divides by N
        # the loss function here must be scaled only by (1 / num_local)
        loss_function = torch.nn.NLLLoss(reduction='sum')
        coef = 1. / (len(training_data_full_sample.dataset) // size)

        self.replace_weights.step(avg_weights, self.device)
        self.model.train()
        grads = [torch.zeros(size=p.shape).to(self.device) for p in self.model.parameters()]
        for batch_idx, (data, target) in enumerate(training_data_full_sample):
            # Print errors (just in case) and zero out the gradient
            torch.autograd.set_detect_anomaly(True)
            self.replace_weights.zero_grad()
            data, target = data.to(self.device).float(), target.to(self.device).long()

            # Forward and backward pass of the model; scale by (1 / N) to line up with average
            out = self.model(data)
            loss = coef * loss_function(out, target)
            loss.backward()

            # Save gradients
            grads = [grads[ind] + p.grad.data.detach().clone().to(self.device) for ind, p in
                     enumerate(self.model.parameters())]

        # Get the average gradient by doing all_reduce and then compute the stationarity violation at the average point
        avg_grads = self.get_average_param(grads)
        stationarity1 = self.regularizer.forward([avg_weights[pa] - avg_grads[pa] for pa in range(self.num_params)],
                                                 self.l1)
        stationarity = numpy.concatenate([avg_weights[pa].detach().cpu().numpy().flatten()
                                          - stationarity1[pa].detach().cpu().numpy().flatten() for pa in
                                          range(self.num_params)])
        # stationarity
        global_norm = numpy.linalg.norm(stationarity, ord=2) ** 2

        # Before sending, also get then number of non-zeros for this agent and this average
        if pre_comm_weights is None:
            _, local_nnz_ratio = self.regularizer.number_non_zeros(local_weights)
            _, nnz_at_average = self.regularizer.number_non_zeros(avg_weights)
        else:
            _, local_nnz_ratio = self.regularizer.number_non_zeros(pre_comm_weights)
            _, nnz_at_average = self.regularizer.number_non_zeros(avg_weights)

        # Perform all-reduce to have sum of local violations, i.e. Frobenius norm of consensus
        array_to_send = numpy.array([local_violation, local_nnz_ratio])
        recv_array = numpy.empty(shape=array_to_send.shape)
        comm.Barrier()
        comm.Allreduce(array_to_send, recv_array)
        comm.Barrier()

        # return consensus, gradient, total optimality, iterate history,
        # local number non-zeros, number nonzeros at everate, and average number of nonzeros
        return recv_array[0], global_norm, recv_array[0] + global_norm, avg_weight_norm, \
               nnz_at_average, (1 / size) * recv_array[1]

    def test(self, weights, testing_data, mode='global'):
        '''Test the data using the average weights'''

        self.replace_weights.zero_grad()
        self.replace_weights.step(weights, self.device)
        self.model.eval()

        # Create separate testing loss for testing data
        loss_function = torch.nn.NLLLoss(reduction='sum')

        # Allocate space for testing loss and accuracy
        test_loss = 0
        correct = 0

        # Do not compute gradient with respect to the testing data
        with torch.no_grad():
            # Loop over testing data
            for data, target in testing_data:
                # Use CUDA if possible
                data, target = data.to(self.device).float(), target.to(self.device).long()

                # Evaluate the model on the testing data
                output = self.model(data)
                test_loss += loss_function(output, target).item()

                # Gather predictions on testing data
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Compute number of testing data points
        num_test_points = int(len(testing_data.dataset) / size)

        # We have two modes of reporting data:
        # 1. We use the AVG weights on all of the data
        # 2. We use the local weights on the local data and THEN compute average
        if mode == 'global':
            # PERFORM ALL REDUCE TO HAVE AVERAGE
            array_to_send = numpy.array([correct, num_test_points, test_loss])
            recv_array = numpy.empty(shape=array_to_send.shape)

            # Barrier
            comm.Barrier()
            comm.Allreduce(array_to_send, recv_array)
            comm.Barrier()

            # Save loss and accuracy
            test_loss = recv_array[2] / recv_array[1]
            testing_accuracy = 100 * recv_array[0] / recv_array[1]

        # Compue local information and then average
        elif mode == 'local':
            # PERFORM ALL REDUCE TO HAVE AVERAGE
            correct /= num_test_points
            test_loss /= num_test_points
            array_to_send = numpy.array([correct, test_loss])
            recv_array = numpy.empty(shape=array_to_send.shape)

            # Barrier
            comm.Barrier()
            comm.Allreduce(array_to_send, recv_array)
            comm.Barrier()

            # Save loss and accuracy
            test_loss = recv_array[1] / size
            testing_accuracy = 100 * recv_array[0] / size
        else:
            sys.exit(f"[ERROR] _ {mode} _ is not a vaild report metric; choose from \'local\' or \'global\' [ERROR]")

        return test_loss, testing_accuracy


if __name__=='__main__':

    # Parse user input
    parser = argparse.ArgumentParser(description='Testing PROXDASAGT on problems from paper.')

    parser.add_argument('--updates', type=int, default=10001, help='Total number of communication rounds.')
    parser.add_argument('--lr', type=float, default=10.0, help='Local learning rate.')
    parser.add_argument('--alpha_base', type=float, default=0.3, help='Moving average rate base')
    parser.add_argument('--l1', type=float, default=1e-4, help='L-1 Regularizer.')
    parser.add_argument('--mini_batch', type=int, default=4, help='Mini-batch size.')
    parser.add_argument('--comm_pattern', type=str, default='ring', choices=['ring', 'random'], help='Communication pattern.')
    parser.add_argument('--comm_round', type=int, default=1, help='m')
    parser.add_argument('--data', type=str, default='a9a', choices=['a9a', 'mnist'], help='Dataset.')
    parser.add_argument('--trial', type=int, default=1, help='Which starting variables to use.')
    parser.add_argument('--step_type', type=str, default='diminishing', choices=('constant', 'diminishing'),
                        help='Diminishing or constant step-size.')
    parser.add_argument('--report', type=int, default=100, help='How often to report criteria.')

    # Create callable argument
    args = parser.parse_args()

    # random seed
    torch.manual_seed(42)

    ###########################
    # a9a data
    if args.data == 'a9a':
        # Subset data to local agent
        num_samples = 32561 // size
        train_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=True),
            batch_size=args.mini_batch, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Load data to be used to compute full gradient with neighbors
        optimality_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=True),
            batch_size=num_samples, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in
             range(int(rank * num_samples), int((rank + 1) * num_samples))]))  # Difference is in number of samples!!

        # Load the testing data
        num_test = 16281 // size
        test_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=False),
            batch_size=num_test, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in range(int(rank * num_test), int((rank + 1) * num_test))]))
    ###########################

    ###########################
    # MNIST data
    else:
        # Create transform for data
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

        # Subset data to local agent
        num_samples = 60000 // size
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                       transform=transform),
            batch_size=args.mini_batch, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Load data to be used to compute full gradient with neighbors
        optimality_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                       transform=transform),
            batch_size=num_samples, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in
             range(int(rank * num_samples), int((rank + 1) * num_samples))]))  # Difference is in number of samples!!

        # Load the testing data
        num_test = 10000 // size
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transform),
            batch_size=num_test, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_test), int((rank + 1) * num_test))]))
    ###########################

    # Load communication matrix and initial weights
    mixing_matrix = torch.tensor(numpy.load(f'mixing_matrices/{args.comm_pattern}_{size}.dat', allow_pickle=True))
    arch_size = 4 if args.data == 'a9a' else 8
    init_weights = [numpy.load(os.path.join(os.getcwd(), f'init_weights/{args.data}/trial{args.trial}/rank{rank}/layer{l}.dat'),
                       allow_pickle=True) for l in range(arch_size)]

    # Print training information
    if rank == 0:
        opening_statement = f' PROXDASAGT on {args.data} '
        print(f"\n{'#' * 75}")
        print('\n' + opening_statement.center(75, ' '))
        print(f'[GRAPH INFO] {size} agents | connectivity = {args.comm_pattern} | \
            rho = {torch.sort(torch.linalg.eig(mixing_matrix)[0].real)[0][size - 2].item()}')        
        print(f'[TRAINING INFO] mini-batch = {args.mini_batch} | learning rate = {args.lr}\n')
        print(f"{'#' * 75}\n")

    # Barrier before training
    comm.Barrier()

    # Declare and train!
    algo_params = {'updates': args.updates, 'lr': args.lr, 'alpha_base': args.alpha_base, 'mini_batch': args.mini_batch, 'report': args.report,
                   'step_type': args.step_type, 'l1': args.l1, 'comm_round': args.comm_round}
    solver = PROXDASAGT(algo_params, mixing_matrix, train_loader, init_weights)
    algo_time = solver.solve(args.updates, optimality_loader, test_loader)

    # Save the information
    method = 'proxdasagt'

    # collect all results in a common folder
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/'))
    except:
        pass
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/plot_results'))
    except:
        pass
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/plot_results/{args.data}'))
    except:
        pass
    path = os.path.join(os.getcwd(), f'results/plot_results/{args.data}')

    # Save information via numpy
    if rank == 0:
        all_results = [solver.testing_loss, solver.testing_accuracy, solver.training_loss, solver.training_accuracy,\
                    solver.testing_loss_local, solver.testing_accuracy_local, solver.training_loss_local, solver.training_accuracy_local,\
                    solver.total_optimality, solver.consensus_violation, solver.norm_hist, solver.iterate_norm_hist, solver.total_time,\
                    solver.communication_time, solver.compute_time, solver.nnz_at_avg, solver.avg_nnz]
        all_results = numpy.array(all_results, dtype=object)
        numpy.save(f'{path}/{method}_t_{args.trial}_{args.comm_pattern}_{args.mini_batch}_{args.updates}_lr_{args.lr}.npy', all_results)
    # Barrier at end so all agents stop this script before moving on
    comm.Barrier()

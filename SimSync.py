# Xihang Yu
# 07/29/2023
# regularized SIM-Sync

import numpy as np
from scipy.linalg import svd
from utils.utils import sorteig,project2SO3
from collections import defaultdict
import sys
import mosek
import copy


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def SimSyncReg(N, edges, pointclouds, scale_gt=None, Weights = None, reg_lambda=1000):


    weights_deepcopy = copy.deepcopy(Weights)
    if Weights is None:
        Weights = defaultdict()
        for idx in range(len(edges)):
            nij = pointclouds[idx].shape[1]
            Weights[edges[idx]] = np.ones(nij)
    else:
        Weights = defaultdict()
        start = 0
        for idx in range(len(edges)):
            nij = pointclouds[idx].shape[1]
            Weights[edges[idx]] = weights_deepcopy[start: start+nij]
            start = start + nij
        assert start == len(weights_deepcopy)

    # Create a task object and attach log stream printer
    with mosek.Task() as task:
        task.set_Stream(mosek.streamtype.log, streamprinter)

        ############################
        ## C matrix lost function ##
        ############################
        Q1 = np.zeros((N, N))
        Q2 = np.zeros((3*N, 3*N))
        V = np.zeros((3*N, N))

        for idx in range(len(edges)):
            edge = edges[idx]
            frame1 = edge[0]
            frame2 = edge[1]
            pointcloud_pair = pointclouds[idx]
            weights_edge_idx = Weights[edges[idx]]
            # Pi: 3,nij
            Pi = np.sqrt(weights_edge_idx) * pointcloud_pair[0:3]
            Pj = np.sqrt(weights_edge_idx) * pointcloud_pair[3:6]
            assert Pi.shape[1] != 0 and Pj.shape[1] != 0, "No correspondences found!"
            assert Pi.shape[0] == 3 and Pj.shape[0] == 3 and Pi.shape[1] == Pj.shape[1]
            nij = pointcloud_pair.shape[1]

            # (frame1,frame2) pair
            start_frame1 = 3*frame1
            end_frame1 = 3*frame1+3
            start_frame2 = 3*frame2
            end_frame2 = 3*frame2+3
            sum_weights_edge_idx = np.sum(weights_edge_idx)

            Q1[frame1, frame1] += sum_weights_edge_idx
            Q1[frame2, frame2] += sum_weights_edge_idx
            Q1[frame1, frame2] -= sum_weights_edge_idx
            Q1[frame2, frame1] -= sum_weights_edge_idx

            PjiT = Pj @ Pi.T
            Q2[start_frame1:end_frame1, start_frame1:end_frame1] += Pi @ Pi.T
            Q2[start_frame2:end_frame2, start_frame1:end_frame1] -= PjiT
            Q2[start_frame1:end_frame1, start_frame2:end_frame2] -= PjiT.T
            Q2[start_frame2:end_frame2, start_frame2:end_frame2] += Pj @ Pj.T

            Pi_sum = np.sum(Pi*np.sqrt(weights_edge_idx),axis=1)
            Pj_sum = np.sum(Pj*np.sqrt(weights_edge_idx),axis=1)
            V[start_frame1:end_frame1, frame1] += Pi_sum
            V[start_frame1:end_frame1, frame2] -= Pi_sum
            V[start_frame2:end_frame2, frame1] -= Pj_sum
            V[start_frame2:end_frame2, frame2] += Pj_sum

        Q1bar = Q1[:,1:]
        # Calculate the Cholesky decomposition
        Q1barTQ1bar = Q1bar.T @ Q1bar
        L = np.linalg.cholesky(Q1barTQ1bar)
        Y = np.linalg.solve(L, -Q1bar.T @ V.T)
        Abar = np.linalg.solve(L.T, Y)

        # Abar, _, _, _ = -np.linalg.lstsq(Q1bar, V.T, rcond=None)
        rtotbar = np.kron(Abar, np.eye(3))
        A = np.zeros((N,3*N))
        A[1:,:] = Abar

        C = A.T @ Q1 @ A + 2 * V @ A + Q2

        assert np.linalg.norm(C - C.T) < 1e-3



        ######################################
        ###### Define Objective Function #####
        ######################################

        # Define variable
        BARVARDIM = [3*N]
        for i in range(N-1):
            BARVARDIM.append(2)
        task.appendbarvars(BARVARDIM)    

        # Define first objective value C1
        barci = []
        barcj = []
        barcval = []
        for i in range(3*N):
            for j in range(0,i+1):
                barci.append(i)
                barcj.append(j)
                barcval.append(C[i,j])
        symc = task.appendsparsesymmat(BARVARDIM[0],
                                       barci,
                                       barcj,
                                       barcval)
        task.putbarcj(0, [symc], [1.0])

        # Define second to N objective values B(1)-B(N-1)
        for i in range(1, N):
            barbi = [1]
            barbj = [1]
            barbval = [reg_lambda]

            symc = task.appendsparsesymmat(BARVARDIM[i],
                                        barbi,
                                        barbj,
                                        barbval)
            task.putbarcj(i, [symc], [1.0])


        ############################
        #### Define Constraints ####
        ############################
        # Set bounds for first sets of constraints (Diagonal constraints)
        bkc = []
        blc = []
        buc = []
        # Upper left identity 
        for i in range(3):
            bkc.append(mosek.boundkey.fx)
            blc.append(1)
            buc.append(1)
        for i in range(3):
            bkc.append(mosek.boundkey.fx)
            blc.append(0)
            buc.append(0)
        # Diagonal scaled identity
        if scale_gt is None:
            for i in range(1, N):
                for j in range(5):
                    bkc.append(mosek.boundkey.fx)
                    # Bound values for constraints
                    blc.append(0)
                    buc.append(0)
        else:
            for i in range(1, N):
                for j in range(3):
                    bkc.append(mosek.boundkey.fx)
                    # Bound values for constraints
                    blc.append(scale_gt[i]**2)
                    buc.append(scale_gt[i]**2)            
                for j in range(3):
                    bkc.append(mosek.boundkey.fx)
                    # Bound values for constraints
                    blc.append(0)
                    buc.append(0)   
        # set A matrices
        barai = []
        baraj = []
        baraval = []
        barai_1 = [0]
        baraj_1 = [0]
        baraval_1 = [1]
        barai.append(barai_1)
        baraj.append(baraj_1)
        baraval.append(baraval_1)
        barai_2 = [1]
        baraj_2 = [1]
        baraval_2 = [1]
        barai.append(barai_2)
        baraj.append(baraj_2)
        baraval.append(baraval_2)
        barai_3 = [2]
        baraj_3 = [2]
        baraval_3 = [1]
        barai.append(barai_3)
        baraj.append(baraj_3)
        baraval.append(baraval_3)
        barai_4 = [1]
        baraj_4 = [0]
        baraval_4 = [1]
        barai.append(barai_4)
        baraj.append(baraj_4)
        baraval.append(baraval_4)
        barai_5 = [2]
        baraj_5 = [0]
        baraval_5 = [1]
        barai.append(barai_5)
        baraj.append(baraj_5)
        baraval.append(baraval_5)
        barai_6 = [2]
        baraj_6 = [1]
        baraval_6 = [1]
        barai.append(barai_6)
        baraj.append(baraj_6)
        baraval.append(baraval_6)
        if scale_gt is None:
            for i in range(1, N):
                barai_1 = [3*i, 3*i+1]
                baraj_1 = [3*i, 3*i+1]
                baraval_1 = [1.0, -1.0]
                barai.append(barai_1)
                baraj.append(baraj_1)
                baraval.append(baraval_1)
                barai_2 = [3*i, 3*i+2]
                baraj_2 = [3*i, 3*i+2]
                baraval_2 = [1.0, -1.0]
                barai.append(barai_2)
                baraj.append(baraj_2)
                baraval.append(baraval_2)
                barai_3 = [3*i+1]
                baraj_3 = [3*i]
                baraval_3 = [1.0]
                barai.append(barai_3)
                baraj.append(baraj_3)
                baraval.append(baraval_3)
                barai_4 = [3*i+2]
                baraj_4 = [3*i]
                baraval_4 = [1.0]
                barai.append(barai_4)
                baraj.append(baraj_4)
                baraval.append(baraval_4)
                barai_5 = [3*i+2]
                baraj_5 = [3*i+1]
                baraval_5 = [1.0]
                barai.append(barai_5)
                baraj.append(baraj_5)
                baraval.append(baraval_5)
        else:
            for i in range(1, N):
                barai_1 = [3*i]
                baraj_1 = [3*i]
                baraval_1 = [1.0]
                barai.append(barai_1)
                baraj.append(baraj_1)
                baraval.append(baraval_1)
                barai_2 = [3*i+1]
                baraj_2 = [3*i+1]
                baraval_2 = [1.0]
                barai.append(barai_2)
                baraj.append(baraj_2)
                baraval.append(baraval_2)
                barai_3 = [3*i+2]
                baraj_3 = [3*i+2]
                baraval_3 = [1.0]
                barai.append(barai_3)
                baraj.append(baraj_3)
                baraval.append(baraval_3)
                barai_4 = [3*i+1]
                baraj_4 = [3*i]
                baraval_4 = [1.0]
                barai.append(barai_4)
                baraj.append(baraj_4)
                baraval.append(baraval_4)
                barai_5 = [3*i+2]
                baraj_5 = [3*i]
                baraval_5 = [1.0]
                barai.append(barai_5)
                baraj.append(baraj_5)
                baraval.append(baraval_5)
                barai_6 = [3*i+2]
                baraj_6 = [3*i+1]
                baraval_6 = [1.0]
                barai.append(barai_6)
                baraj.append(baraj_6)
                baraval.append(baraval_6)

        num_main_con = len(bkc)
        task.appendcons(num_main_con+2*(N-1))


        for i in range(num_main_con):
            task.putconbound(i, bkc[i], blc[i], buc[i])
        for i in range(num_main_con):
            symai = task.appendsparsesymmat(BARVARDIM[0],
                                            barai[i],
                                            baraj[i],
                                            baraval[i])
            task.putbaraij(i, 0, [symai], [1.0])

        # Constraints for regularizers (num_main_con to num_main_con+2*(N-1))
        counter_con = num_main_con
        for i in range(1, N):
            # Each regularizer has two constraints
            ## First constraint
            task.putconbound(counter_con, mosek.boundkey.fx, 1, 1)
            syma1 = task.appendsparsesymmat(BARVARDIM[0], #dim
                                            [3*i],
                                            [3*i],
                                            [1])
            task.putbaraij(counter_con, 0, [syma1], [1.0])                                            
            syma2 = task.appendsparsesymmat(2, #dim
                                            [1], # i
                                            [0], # j
                                            [-1]) # value
            task.putbaraij(counter_con, i, [syma2], [1.0])  
            counter_con += 1
            ## Second constraint
            task.putconbound(counter_con, mosek.boundkey.fx, 1, 1)                                          
            syma1 = task.appendsparsesymmat(2, #dim
                                            [0],
                                            [0],
                                            [1])
            task.putbaraij(counter_con, i, [syma1], [1.0])
            counter_con += 1

        #####################
        ### Solve problem ###
        #####################
        task.putobjsense(mosek.objsense.minimize)
        task.optimize()
        task.solutionsummary(mosek.streamtype.msg)

        solsta = task.getsolsta(mosek.soltype.itr)

        if (solsta == mosek.solsta.optimal):
            barx = task.getbarxj(mosek.soltype.itr, 0)

            # The operator >> denotes matrix inequality.
            def reconstruct_symmetric_matrix(values, n):
                X = np.zeros((n, n))
                idx = 0
                for j in range(n):
                    for i in range(j, n):
                        X[i, j] = values[idx]
                        if i != j:
                            X[j, i] = values[idx]
                        idx += 1
                return X
                
            X_sol = reconstruct_symmetric_matrix(barx, 3*N)
            
            t_list = []
            for i in range(1, N):
                barx = task.getbarxj(mosek.soltype.itr, i)
                Xi_sol = reconstruct_symmetric_matrix(barx, 2)
                t_list.append(Xi_sol[1,1])
            t_list = np.array(t_list)

            f_sdp = task.getprimalobj(mosek.soltype.itr) # What is this?

            _, singular_values, _ = svd(X_sol)    
            rank_moment = np.sum(singular_values > 1e-3)

            V, eigval = sorteig(X_sol)
            R = (V[:,0:3] @ np.diag(np.sqrt(eigval[0:3]))).T
            R = R[0:3,0:3].T @ R

            # assert np.linalg.norm(X_sol - R.T @ R ) < 1e-1

            # assert np.linalg.norm(R[0:3,0:3] - np.eye(3)) < 1e-2


            # Initialize variables
            s_est = np.zeros(N)
            R_est = np.zeros((3*N,3)) # R_est size: 3*N, 3

            s_est[0] = 1
            R_est[0:3,0:3] = np.eye(3)
            sR_est = np.zeros((3,3*N))
            sR_est[0:3,0:3] = np.eye(3)
            for i in range(1,N):
                si = np.linalg.norm(R[:,3*i:3*i+3]) / np.sqrt(3)
                s_est[i] = si
                R_est_i = project2SO3(R[:,3*i:3*i+3]/si)
                R_est[3*i:3*(i+1), :] = R_est_i
                sR_est[:, 3*i:3*(i+1)] = si * R_est_i
            
            sr_est = sR_est.flatten('F')
            # recover translation from estimated rotation and scale
            tbar_est = rtotbar @ sr_est
            reshaped_tbar_est = np.reshape(tbar_est,(3,N-1), order='F')
            t1 = np.zeros((3,1))
            t_est = np.hstack((t1,reshaped_tbar_est))
            # f_est = sr_est.T @ np.kron(C, np.eye(3)) @ sr_est
            f_est = sr_est.T @ np.kron(C, np.eye(3)) @ sr_est + np.sum(t_list)*reg_lambda
            absDualityGap = abs(f_est - f_sdp)
            relDualityGap = absDualityGap / (1 + abs(f_sdp) + abs(f_est))

            solution  = defaultdict()
            solution["type"] = 'SimSync'
            solution["f_est"] = f_est
            solution["f_sdp"] = f_sdp
            solution["absDualityGap"] = absDualityGap
            solution["relDualityGap"] = relDualityGap
            solution["R_est"] = R_est
            solution["t_est"] = t_est
            solution["s_est"] = s_est
            solution["rank_moment"] = rank_moment

            # Print result.
            print("f_sdp: {}, f_est: {}, absDualityGap: {}, relDualityGap: {}, rank_moment = {}".format(f_sdp, f_est, absDualityGap,relDualityGap, rank_moment))

            solution["f_val"] = f_est # f_val is equal to f_est


            # Residues
            residues = defaultdict()
            for idx in range(len(edges)):
                nij = pointclouds[idx].shape[1]
                residues[edges[idx]] = np.zeros((3,nij))
                i,j = edges[idx]
                pointcloud_pair = pointclouds[idx]
                Pi = pointcloud_pair[0:3]
                Pj = pointcloud_pair[3:6]

                Rsi = sR_est[:, 3*i:3*i+3]
                Rsj = sR_est[:, 3*j:3*j+3]
                ti = t_est[:, i]
                tj = t_est[:, j]
                for k in range(nij):
                    residues[edges[idx]][:, k] = (Rsi @ Pi[:, k] + ti - Rsj @ Pj[:, k] - tj).reshape(-1,)

            # Residuals
            residuals = defaultdict()
            for idx in range(len(edges)):
                nij = pointclouds[idx].shape[1]
                residuals[edges[idx]] = np.zeros(nij)
                for k in range(nij):
                    residuals[edges[idx]][k] = np.linalg.norm(residues[edges[idx]][:, k]) ## debug

            solution["residuals"] = residuals
            solution["residues"] = residues

            return solution
        elif (solsta == mosek.solsta.dual_infeas_cer or
            solsta == mosek.solsta.prim_infeas_cer):
            print("Primal or dual infeasibility certificate found.\n")
        elif solsta == mosek.solsta.unknown:
            print("Unknown solution status")
        else:
            print("Other solution status")
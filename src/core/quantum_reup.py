from qiskit.circuit import ParameterVector, QuantumCircuit

from helpers.helper import (
    pair_inputs_and_weights_single_qubit,
    pair_inputs_and_weights_multi_qubit,
    sample_fx
)


def create_quantum_reup_circuit(
        num_inputs=None,
        inputs=None,
        weights=None,
        args_dict=None
):

    if args_dict['q_num_qubit'] == 1:

        # create a parameterized circuit
        qc = QuantumCircuit(args_dict['q_num_qubit'])

        blocks = pair_inputs_and_weights_single_qubit(inputs=inputs, weights=weights)

        for b in range(len(blocks)):
            if b % 3 == 0:
                qc.rx(blocks[b][0] * blocks[b][1] + blocks[b][2], 0)
            elif b % 3 == 1:
                qc.ry(blocks[b][0] * blocks[b][1] + blocks[b][2], 0)
            else:
                qc.rz(blocks[b][0] * blocks[b][1] + blocks[b][2], 0)
            if blocks[b][0].name == inputs[-1].name:
                qc.barrier()
    else:
        # create a parameterized circuit
        qc = QuantumCircuit(args_dict["q_num_qubit"])

        qubit_pairs = [[i, i+1] for i in range(args_dict["q_num_qubit"] - 1)]
        
        number = int((num_inputs*args_dict["q_num_layers"])/ args_dict["q_num_qubit"])
        blocks = pair_inputs_and_weights_multi_qubit(
            inputs=inputs, 
            weights=weights,
            q_num_layers=args_dict["q_num_layers"], 
            q_num_qubit=args_dict["q_num_qubit"], 
            q_reup_method=args_dict["q_reup_method"]
        )
        circuit_list = sample_fx(blocks,number)

        if args_dict["q_reup_method"] == "symmetrical":
            entanglement_index = 0
            for b in range(len(blocks)):
                if b % 3 == 0:
                    for i in range(len(blocks[b])):
                        qc.rx(blocks[b][i][0] * blocks[b][i][1] + blocks[b][i][2],blocks[b][i][3])
                elif b % 3 == 1:
                    for i in range(len(blocks[b])):
                        qc.ry(blocks[b][i][0] * blocks[b][i][1] + blocks[b][i][2],blocks[b][i][3])
                else:
                    for i in range(len(blocks[b])):
                        qc.rz(blocks[b][i][0] * blocks[b][i][1] + blocks[b][i][2],blocks[b][i][3])

                if blocks[b][i][0].name == inputs[-1].name:
                    qc.barrier()
                    if args_dict["q_entanglement"]:
                        if args_dict["q_num_qubit"] == 2:
                            if b != (len(blocks)-1): # ensure it is not the last layer
                                qc.cz(0, 1)
                                qc.barrier()
                        elif args_dict["q_num_qubit"] == 3:
                            if entanglement_index % 2 == 0:
                                if b != (len(blocks)-1): # ensure it is not the last layer
                                    qc.cz(0, 1)
                                    qc.cz(1, 2)
                                    qc.barrier()
                            else:
                                if b != (len(blocks)-1): # ensure it is not the last layer
                                    qc.cz(1, 2)
                                    qc.cz(0, 1)
                                    qc.barrier()

                        elif args_dict["q_num_qubit"] == 4:
                            if entanglement_index % 2 == 0:
                                if b != (len(blocks)-1): # ensure it is not the last layer
                                    qc.cz(0,1)
                                    qc.cz(2,3)
                                    qc.barrier()
                            else:
                                if b != (len(blocks)-1): # ensure it is not the last layer
                                    qc.cz(1,2)
                                    qc.cz(0,3)
                                    qc.barrier()
                        elif args_dict["q_num_qubit"] == 5:
                            if b != (len(blocks)-1): # ensure it is not the last layer
                                qc.cz(0,1)
                                qc.cz(1,2)
                                qc.cz(2,3)
                                qc.cz(3,4)
                                qc.cz(0,4)
                                qc.barrier()

                        entanglement_index += 1

        elif args_dict["q_reup_method"] == "asymmetrical":
            entanglement_index = 0
            for b in range(len(blocks)):
                if b % 3 == 0:
                    for i in range(len(blocks[b])):
                        qc.rx(blocks[b][i][0] * blocks[b][i][1] + blocks[b][i][2],blocks[b][i][3])
                elif b % 3 == 1:
                    for i in range(len(blocks[b])):
                        qc.ry(blocks[b][i][0] * blocks[b][i][1] + blocks[b][i][2],blocks[b][i][3])
                else:
                    for i in range(len(blocks[b])):
                        qc.rz(blocks[b][i][0] * blocks[b][i][1] + blocks[b][i][2],blocks[b][i][3])
                
                if blocks[b][i][0].name == inputs[-1].name:
                    qc.barrier()
                    if args_dict["q_entanglement"]:
                        if args_dict["q_num_qubit"] == 2:
                            if b != (len(blocks)-1): # ensure it is not the last layer
                                qc.cz(0, 1)
                                qc.barrier()
                        elif args_dict["q_num_qubit"] == 3:
                            if entanglement_index % 2 == 0:
                                if b != (len(blocks)-1): # ensure it is not the last layer
                                    qc.cz(0, 1)
                                    qc.cz(1, 2)
                                    qc.barrier()
                            else:
                                if b != (len(blocks)-1): # ensure it is not the last layer
                                    qc.cz(1, 2)
                                    qc.cz(0, 1)
                                    qc.barrier()

                        elif args_dict["q_num_qubit"] == 4:
                            if entanglement_index % 2 == 0:
                                if b != (len(blocks)-1): # ensure it is not the last layer
                                    qc.cz(0,1)
                                    qc.cz(2,3)
                                    qc.barrier()
                            else:
                                if b != (len(blocks)-1): # ensure it is not the last layer
                                    qc.cz(1,2)
                                    qc.cz(0,3)
                                    qc.barrier()
                        elif args_dict["q_num_qubit"] == 5:
                            if b != (len(blocks)-1): # ensure it is not the last layer
                                qc.cz(0,1)
                                qc.cz(1,2)
                                qc.cz(2,3)
                                qc.cz(3,4)
                                qc.cz(0,4)
                                qc.barrier()

                        entanglement_index += 1

                        entanglement_index += 1

        else:
            raise Exception("Error: Must specify qnn-reup strategy with q_reup_method")
    
    return qc
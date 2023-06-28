p2p_matrix: p2p_matrix.cc
	nvcc -lmpi -lnccl p2p_matrix.cc -o p2p_matrix

clean:
	rm p2pmatrix


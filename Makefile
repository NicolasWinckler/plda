CC=g++
MPICC=mpicxx

#CFLAGS=-O3 -std=c++11 -Wall -Wno-sign-compare
CFLAGS=-O3 -Wall -Wno-sign-compare
OBJ_PATH = ./obj

all: lda infer mpi_lda lda_with_doc mpi_lda_with_doc

clean:
	rm -rf $(OBJ_PATH)
	rm -f lda mpi_lda infer lda_with_doc mpi_lda_with_doc

OBJ_SRCS := cmd_flags.cc common.cc document.cc model.cc accumulative_model.cc sampler.cc
ALL_OBJ = $(patsubst %.cc, %.o, $(OBJ_SRCS))
OBJ = $(addprefix $(OBJ_PATH)/, $(ALL_OBJ))

$(OBJ_PATH)/%.o: %.cc
	@ mkdir -p $(OBJ_PATH) 
	$(CC) -c $(CFLAGS) $< -o $@

lda: lda.cc $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $< -o $@

infer: infer.cc $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $< -o $@

mpi_lda: mpi_lda.cc $(OBJ)
	$(MPICC) $(CFLAGS) $(OBJ) $< -o $@

lda_with_doc: lda_with_doc.cc $(OBJ)
	$(MPICC) $(CFLAGS) $(OBJ) $< -o $@

mpi_lda_with_doc: mpi_lda_with_doc.cc $(OBJ)
	$(MPICC) $(CFLAGS) $(OBJ) $< -o $@


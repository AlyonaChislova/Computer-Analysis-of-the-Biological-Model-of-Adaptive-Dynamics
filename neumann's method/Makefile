CC=g++
CFLAGS=-Wall -O3 -Ofast -DDEBUG
LIBS=-lm -lfftw3

SRC=solver.cpp problem.cpp vector_handler.cpp
OBJ=$(SRC:%.cpp=%.o)

%.o: %.cpp %.hpp
	$(CC) $(CFLAGS) $(INCDIR) -c $< -o $@

exec: main.cpp $(OBJ)
	$(CC) $(CFLAGS) $(INCDIR) $(LDFLAGS) $^ $(LIBS) -o $@

clean:
	rm -f *.o
	rm -f exec

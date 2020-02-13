CXX		  := g++
CXX_FLAGS := -Wall -Wextra  -ggdb 

BIN		:= bin
SRC		:= src
INCLUDE	:= include
LIB		:= lib


LIBRARIES	:=-lv4l2 -lm
EXECUTABLE	:= main

OPENCV_LIBS = $(shell pkg-config --libs opencv4)
OPENCV_INCLUDE = $(shell pkg-config --cflags opencv4)

all: $(BIN)/$(EXECUTABLE)

run: clean all
	clear
	./$(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)/*.cpp
	$(CXX) $(CXX_FLAGS) -DDEBUG -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES) $(OPENCV_LIBS) $(OPENCV_INCLUDE)

clean:
	-rm $(BIN)/*

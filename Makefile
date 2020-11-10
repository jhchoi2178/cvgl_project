CXX=g++
CXXFLAGS=`pkg-config opencv4 --cflags`
LDLIBS= -lglut -lGL -lGLU `pkg-config opencv4 --libs`
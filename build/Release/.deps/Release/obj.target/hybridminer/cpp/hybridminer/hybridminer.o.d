cmd_Release/obj.target/hybridminer/cpp/hybridminer/hybridminer.o := g++ '-DNODE_GYP_MODULE_NAME=hybridminer' '-DUSING_UV_SHARED=1' '-DUSING_V8_SHARED=1' '-DV8_DEPRECATION_WARNINGS=1' '-D_LARGEFILE_SOURCE' '-D_FILE_OFFSET_BITS=64' '-DBUILDING_NODE_EXTENSION' -I/home/andy/.node-gyp/8.9.4/include/node -I/home/andy/.node-gyp/8.9.4/src -I/home/andy/.node-gyp/8.9.4/deps/uv/include -I/home/andy/.node-gyp/8.9.4/deps/v8/include -I../node_modules/nan -I/usr/local/include  -fPIC -pthread -Wall -Wextra -Wno-unused-parameter -m64 -O3 -fno-omit-frame-pointer -march=native -O3 -std=c++11 -fno-rtti -fno-exceptions -std=gnu++0x -MMD -MF ./Release/.deps/Release/obj.target/hybridminer/cpp/hybridminer/hybridminer.o.d.raw   -c -o Release/obj.target/hybridminer/cpp/hybridminer/hybridminer.o ../cpp/hybridminer/hybridminer.cpp
Release/obj.target/hybridminer/cpp/hybridminer/hybridminer.o: \
 ../cpp/hybridminer/hybridminer.cpp ../cpp/hybridminer/hybridminer.h \
 ../cpp/hybridminer/cpusolver.h ../cpp/hybridminer/cudasolver.h
../cpp/hybridminer/hybridminer.cpp:
../cpp/hybridminer/hybridminer.h:
../cpp/hybridminer/cpusolver.h:
../cpp/hybridminer/cudasolver.h:

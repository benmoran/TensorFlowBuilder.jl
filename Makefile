
.PHONY:	all clean setup generate install

GEN_PATH := ./generated/TensorFlow

all:	clean setup generate

clean:
	rm -rf ${GEN_PATH}

setup:	src/CoreTypes.jl src/dtypes.jl src/types.jl
	mkdir -p ${GEN_PATH}/src
	mkdir -p ${GEN_PATH}/test
	cp src/TensorFlow.jl src/API.jl src/CoreTypes.jl src/dtypes.jl src/types.jl ${GEN_PATH}/src/

generate:	setup
	julia src/generate_api.jl ${GEN_PATH} > generate_api.log 2>&1

install:
	rsync -va --exclude TensorFlow.jl ${GEN_PATH}/ ../TensorFlow/

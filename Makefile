
.PHONY:	clean setup generate install

GEN_PATH := ./generated/TensorFlow

clean:
	rm -rf ${GEN_PATH}

setup:	src/CoreTypes.jl src/dtypes.jl src/types.jl
	mkdir -p ${GEN_PATH}/src
	mkdir -p ${GEN_PATH}/test
	cp src/CoreTypes.jl src/dtypes.jl src/types.jl ${GEN_PATH}/src/

generate:
	julia src/generate_api.jl ${GEN_PATH}

install:
	echo rsync -v ${GEN_PATH} ../TensorFlow

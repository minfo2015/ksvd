################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/addnoise_function.cpp \
../src/ksvd.cpp \
../src/lib_ormp.cpp \
../src/lib_svd.cpp \
../src/utilities.cpp 

C_SRCS += \
../src/io_png.c \
../src/mt19937ar.c 

CU_SRCS += \
../src/Utilities.cu \
../src/main.cu 

CU_DEPS += \
./src/Utilities.d \
./src/main.d 

OBJS += \
./src/Utilities.o \
./src/addnoise_function.o \
./src/io_png.o \
./src/ksvd.o \
./src/lib_ormp.o \
./src/lib_svd.o \
./src/main.o \
./src/mt19937ar.o \
./src/utilities.o 

C_DEPS += \
./src/io_png.d \
./src/mt19937ar.d 

CPP_DEPS += \
./src/addnoise_function.d \
./src/ksvd.d \
./src/lib_ormp.d \
./src/lib_svd.d \
./src/utilities.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I"/0_Simple" -I"/common/inc" -G -g -O0 -gencode arch=compute_20,code=sm_20  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I"/0_Simple" -I"/common/inc" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I"/0_Simple" -I"/common/inc" -G -g -O0 -gencode arch=compute_20,code=sm_20  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I"/0_Simple" -I"/common/inc" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I"/0_Simple" -I"/common/inc" -G -g -O0 -gencode arch=compute_20,code=sm_20  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I"/0_Simple" -I"/common/inc" -G -g -O0 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



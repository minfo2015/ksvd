################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/vectorAdd.cu 

CU_DEPS += \
./src/vectorAdd.d 

OBJS += \
./src/vectorAdd.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I"/usr/local/cuda-7.5/samples/0_Simple" -I"/usr/local/cuda-7.5/samples/common/inc" -I"/home/asr/cuda-workspace/SVD_Test" -G -g -O0 -gencode arch=compute_20,code=sm_20  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I"/usr/local/cuda-7.5/samples/0_Simple" -I"/usr/local/cuda-7.5/samples/common/inc" -I"/home/asr/cuda-workspace/SVD_Test" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



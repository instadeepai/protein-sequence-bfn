#################
# CPU, TPU or GPU. If you enter an incorret option, it will default to CPU-only
ACCELERATOR = TPU

# variables
WORK_DIR = $(PWD)
USER_ID = $$(id -u)
GROUP_ID = $$(id -g)

DOCKER_BUILD_FLAGS = --build-arg USER_ID=$(USER_ID) \
	--build-arg GROUP_ID=$(GROUP_ID) \

ifeq ($(ACCELERATOR), GPU)
	DOCKER_BUILD_FLAGS += --build-arg BASE_IMAGE="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04"
endif

ifeq ($(ACCELERATOR), TPU)
	DOCKER_BUILD_FLAGS +=  --build-arg BASE_IMAGE="ubuntu:20.04"
endif

ifeq ($(ACCELERATOR), CPU)
	DOCKER_BUILD_FLAGS +=  --build-arg BASE_IMAGE="ubuntu:20.04"
endif

DOCKER_RUN_FLAGS_CPU = --rm \
	--shm-size=1024m \
	-v $(WORK_DIR):/app

DOCKER_RUN_FLAGS_GPU = ${DOCKER_RUN_FLAGS_CPU} --gpus all 

DOCKER_RUN_FLAGS_TPU = --rm --user root --privileged \
	-v $(WORK_DIR):/app

# Select appropriate run flags based on accelerator
ifeq ($(ACCELERATOR), CPU)
	DOCKER_RUN_FLAGS = $(DOCKER_RUN_FLAGS_CPU)
else ifeq ($(ACCELERATOR), GPU)
	DOCKER_RUN_FLAGS = $(DOCKER_RUN_FLAGS_GPU)
else ifeq ($(ACCELERATOR), TPU)
	DOCKER_RUN_FLAGS = $(DOCKER_RUN_FLAGS_TPU)
endif

# image + container name
DOCKER_IMAGE_NAME = protbfn
DOCKER_CONTAINER_NAME = protbfn_container


.PHONY: build
build:
	sudo docker build -t $(DOCKER_IMAGE_NAME) -f Dockerfile . \
		$(DOCKER_BUILD_FLAGS) --build-arg ACCELERATOR=$(ACCELERATOR) \

.PHONY: sample
sample:
	sudo docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_NAME) python sample.py $(RUN_ARGS)

.PHONY: inpaint
inpaint:
	sudo docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_NAME) python inpaint.py $(RUN_ARGS)
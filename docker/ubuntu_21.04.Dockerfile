FROM ubuntu:21.04

RUN apt-get upgrade
RUN apt-get update
RUN apt-get -y install apt-utils software-properties-common

RUN apt-get -y install gcc-9 g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 800 --slave /usr/bin/g++ g++ /usr/bin/g++-9

RUN apt-get -y install curl
RUN ln -s /usr/lib/x86_64-linux-gnu/libz3.so.4 /usr/lib/libz3.so.4.8
RUN curl -SL https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz | tar -xJC .
RUN mv clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04 /usr/clang

RUN apt-get -y install wget xz-utils libz3-4
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.tar.gz
RUN tar -zxf cmake-3.22.1-linux-x86_64.tar.gz
RUN mv cmake-3.22.1-linux-x86_64 /usr/cmake
RUN rm cmake-3.22.1-linux-x86_64.tar.gz

RUN apt-get -y install make
RUN apt-get -y install libtinfo5

RUN apt-get install -y bison flex

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install tzdata

RUN apt-get update \
  && apt-get install -y ssh \
      build-essential \
      gcc \
      g++ \
      gdb \
      clang \
      make \
      ninja-build \
      cmake \
      autoconf \
      automake \
      locales-all \
      dos2unix \
      rsync \
      tar \
      python \
  && apt-get clean

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

RUN useradd -m user \
  && yes password | passwd user

RUN usermod -s /bin/bash user

ENV LIBRARY_PATH $LIBRARY_PATH:/usr/clang/lib
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/clang/lib
ENV PATH $PATH:/usr/clang/bin:/usr/cmake/bin

WORKDIR /code
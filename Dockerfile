FROM ubuntu:16.04

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install g++ valgrind make libopenblas-dev nano -y

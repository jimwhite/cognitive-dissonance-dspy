FROM acl2-jupyter:latest

RUN sudo apt update && sudo apt install -y coq

EXPOSE 1234


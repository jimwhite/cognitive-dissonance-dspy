# ARG BASE_IMAGE=ghcr.io/jimwhite/acl2-jupyter:latest
ARG BASE_IMAGE=acl2-jupyter:latest

FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/jimwhite/cognitively-guided-reasoning"

USER root
RUN apt update && apt install -y coq

COPY . ${HOME}/cognitively-guided-reasoning
WORKDIR ${HOME}/cognitively-guided-reasoning
RUN make install 
RUN chown -R jovyan:users ${HOME}/cognitively-guided-reasoning
USER jovyan
RUN make test

# Cassandra with jolokia JVM agent

FROM cassandra:3.11

ENV JOLOKIA_VERSION 1.7.2

RUN \
    wget -O - https://github.com/jolokia/jolokia/releases/download/v$JOLOKIA_VERSION/jolokia-$JOLOKIA_VERSION-bin.tar.gz | \
    tar -z --extract --file - jolokia-$JOLOKIA_VERSION/agents/jolokia-jvm.jar; \
    mv jolokia-$JOLOKIA_VERSION/agents/jolokia-jvm.jar $CASSANDRA_HOME/lib/; \
    rm -rf jolokia-$JOLOKIA_VERSION


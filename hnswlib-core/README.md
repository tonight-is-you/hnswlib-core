[![Maven Central](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-core/badge.svg)](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-core) [![Javadoc](http://javadoc-badge.appspot.com/com.github.jelmerk/hnswlib-core.svg?label=javadoc)](http://javadoc-badge.appspot.com/com.github.jelmerk/hnswlib-core)


hnswlib-core
============

Core hnsw library.


Example usage
-------------

```bash
GoodsFile=$1
topK=$2
nb=$3
ef=$4
embDim=$5
OutPutFile=$6
UserEmb=$7

java -cp hnswlib-core-0.0.46.jar:eclipse-collections-10.2.0.jar:eclipse-collections-api-10.2.0.jar com.github.jelmerk.knn.hnsw.EmbText \
    $1 $2 $3 $4 $5 $6 $7 
```

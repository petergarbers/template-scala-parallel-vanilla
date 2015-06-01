package org.template.vanilla

import io.prediction.controller.PDataSource
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.EmptyActualResult
import io.prediction.controller.Params
import io.prediction.data.storage.Event
import io.prediction.data.store.PEventStore

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import grizzled.slf4j.Logger

case class DataSourceParams(appName: String) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  override
  def readTraining(sc: SparkContext): TrainingData = {

    // read all events of EVENT involving ENTITY_TYPE and TARGET_ENTITY_TYPE
    val observationRDD: RDD[Observation] = PEventStore.find(
      appName = "MyApp1", // dsp.appName
      entityType = Some("item"))(sc).map{ element =>
        val labels = element.properties.get[List[String]]("labels").map(_.toDouble).toArray
        val features = element.properties.get[List[String]]("features").map(_.toDouble).toArray
        Observation(labels, features)
      }

    new TrainingData(observationRDD)
  }
}

case class Observation(labels : Array[Double], features : Array[Double]) extends Serializable

class TrainingData(val observations: RDD[Observation]) extends Serializable {
  override def toString = {
    s"observations [${observations.count()}] (${observations.take(2).toList}...)"
  }
}

package org.template.vanilla

import io.prediction.controller.PPreparator

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    val allLabels = trainingData.observations.flatMap(td => td.labels).distinct().collect()
    val labelTDMap: scala.collection.immutable.Map[Double, RDD[LabeledPoint]] = allLabels.map { label =>
      val labelPoints =  trainingData.observations.map  { obs =>
        val exists = if (obs.labels.contains(label)) 1.0 else 0.0
        LabeledPoint(exists, Vectors.dense(obs.features))
      }
      (label, labelPoints)
    }.toMap

    new PreparedData(labeledPoints = labelTDMap)
  }
}

class PreparedData(val labeledPoints: Map[Double, RDD[LabeledPoint]]) extends Serializable
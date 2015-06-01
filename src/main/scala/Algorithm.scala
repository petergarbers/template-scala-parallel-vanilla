package org.template.vanilla

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.linalg.Vectors

import grizzled.slf4j.Logger

case class RandomForestAlgorithmParams(numClasses: Int,
                                       numTrees: Int,
                                       featureSubsetStrategy: String,
                                       impurity: String,
                                       maxDepth: Int,
                                       maxBins: Int) extends Params

class RandomForestAlgorithm(val ap: RandomForestAlgorithmParams)
  // extends PAlgorithm if Model contains RDD[]
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
//    require(data.labeledPoints.take(1).nonEmpty,
//      s"RDD[labeledPoints] in PreparedData cannot be empty. ${data.labeledPoints.take(1).length}")

//    val numClasses = 2
//    val categoricalFeaturesInfo = Map[Int, Int]()
//    val numTrees = 3 // Use more in practice.
//    val featureSubsetStrategy = "auto" // Let the algorithm choose.
//    val impurity = "gini"
//    val maxDepth = 4
//    val maxBins = 32

    val categoricalFeaturesInfo = Map[Int, Int]()
    val models: Map[Double, RandomForestModel] = data.labeledPoints.map{ case (label, rdd) =>
      (label, RandomForest.trainClassifier(rdd, ap.numClasses, categoricalFeaturesInfo, ap.numTrees, ap.featureSubsetStrategy, ap.impurity, ap.maxDepth, ap.maxBins))
//      (label, RandomForest.trainClassifier(rdd, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins))
    }

    new Model(models = models)
  }

  def predict(models: Model, query: Query): PredictedResult = {
    val predictedLabels: Set[Double] = models.models.map{ case (label, model) =>
      (label, model.predict(Vectors.dense(query.features)))
    }.filter( pl => pl._2 == 1.0).keys.toSet

    PredictedResult(results = predictedLabels)
  }
}

class Model(val models: Map[Double, RandomForestModel]) extends Serializable {
  override def toString = s"mc=${models}"
}

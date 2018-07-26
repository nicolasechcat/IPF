package org.apache.spark.mllib.feature

{

import java.io.Serializable

import org.apache.spark.SparkContext

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.RandomForest

class IPF_RF (val data: RDD[LabeledPoint], val sc: SparkContext, val k: Int = 5, 
        val deletedInstances: Double = 0.1, val maxIterations: Int = 5, 
        val threshold: Double = 0.5, 
        val numtrees: Int = 50, val seed: Int = 0) extends Serializable  {

    def runFilter(): RDD[LabeledPoint] = {

        val labelsLength = data.distinct().count.toInt
        val maxError = sc.broadcast(k * threshold)
        val totalInstances = data.count
        val deletedInstStop = totalInstances * deletedInstances

        val filteredData = data
        var iterations = 0
        var deletedInst: Long = 0

        def loop (loopData: RDD[LabeledPoint]): RDD[LabeledPoint] = {
            
            val cvdat = MLUtils.kFold(loopData, k, seed)
            val models = for ((tra, _) <- cvdat)
                    RandomForest.trainClassifier(tra, labelsLength, Map[Int, Int](), numtrees, 
                        "all", "gini", 10, 32, seed = seed)

            val broadcastedModels = sc.broadcast(models)

            val revised = loopData.filter {
                case d:
                    val err = for (model <- broadcastedModels.value) 
                        (if (model.predict(d.features) == d.label) 0 else 1). sum

                    err >= maxError.value            
            }

            return revised
        } 

        while ((iterations < maxIterations) && (deletedInst < deletedInstStop)) {
            filteredData = loop(filteredData)
            deletedInst = totalInstances - filteredData.count
            iterations += 1
        }
        
        return filteredData
    }
}
}

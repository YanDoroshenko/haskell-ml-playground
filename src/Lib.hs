{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Lib
    ( someFunc
    ) where

import Data.Int
import TensorFlow.Core
import qualified TensorFlow.Ops as TO hiding (initializedVariable)
import TensorFlow.Variable
import qualified Data.Vector as V
import TensorFlow.GenOps.Core (square)
import TensorFlow.Minimize (gradientDescent, minimizeWith)
import Control.Monad (replicateM_)
import TensorFlow.Session (run)

runVariable :: V.Vector Float -> V.Vector Float -> IO (Float, Float)
runVariable xInput yInput = runSession $ do
  let xSize = fromIntegral $ V.length xInput
  let ySize = fromIntegral $ V.length yInput
  (w :: Variable Float) <- initializedVariable 0
  (b :: Variable Float) <- initializedVariable 0
  (x :: Tensor Value Float) <- TO.placeholder [xSize]
  let linear_model = (readValue w `TO.mul` x) `TO.add` readValue b
  (y :: Tensor Value Float) <- TO.placeholder [ySize]
  let square_deltas = square (linear_model `TO.sub` y)
  let loss = TO.reduceSum square_deltas
  trainStep <- minimizeWith (gradientDescent 0.01) loss [w,b]
  let trainWithFeeds = \xF yF -> runWithFeeds
          [ feed x xF
          , feed y yF
          ]
          trainStep
  replicateM_ 1000
      (trainWithFeeds (encodeTensorData [xSize] xInput) (encodeTensorData [ySize] yInput))
  (Scalar w_learned, Scalar b_learned) <- run (readValue w, readValue b)
  return (w_learned, b_learned)

someFunc :: IO ()
someFunc = do
  result <- runVariable [0, 1, 2, 3, 4] [1, 3, 5, 7, 9]
  print result

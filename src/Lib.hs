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
import qualified TensorFlow.Minimize as TM
import Control.Monad (replicateM_)
import TensorFlow.Session (run)

learnParametersPolynomial1 :: V.Vector Float -> V.Vector Float -> IO (Float, Float)
learnParametersPolynomial1 xInput yInput = runSession $ do
  let xSize = fromIntegral $ V.length xInput
  let ySize = fromIntegral $ V.length yInput
  (w :: Variable Float) <- initializedVariable 0
  (b :: Variable Float) <- initializedVariable 0
  (x :: Tensor Value Float) <- TO.placeholder [xSize]
  let linear_model = (readValue w `TO.mul` x) `TO.add` readValue b
  (y :: Tensor Value Float) <- TO.placeholder [ySize]
  let square_deltas = square (linear_model `TO.sub` y)
  let loss = TO.reduceSum square_deltas
  trainStep <- TM.minimizeWith (TM.gradientDescent 0.01) loss [w,b]
  let trainWithFeeds = \xF yF -> runWithFeeds
          [ feed x xF
          , feed y yF
          ]
          trainStep
  replicateM_ 1000
      (trainWithFeeds (encodeTensorData [xSize] xInput) (encodeTensorData [ySize] yInput))
  (Scalar w_learned, Scalar b_learned) <- run (readValue w, readValue b)
  return (w_learned, b_learned)

learnParametersPolynomial2 :: V.Vector Float -> V.Vector Float -> IO (Float, Float, Float)
learnParametersPolynomial2 xInput yInput = runSession $ do
  let xSize = fromIntegral $ V.length xInput
  let ySize = fromIntegral $ V.length yInput
  (a :: Variable Float) <- initializedVariable 0
  (b :: Variable Float) <- initializedVariable 0
  (c :: Variable Float) <- initializedVariable 0
  (x :: Tensor Value Float) <- TO.placeholder [xSize]
  let polynomial2_model = (readValue a `TO.mul` (x `TO.mul` x)) `TO.add` (readValue b `TO.mul` x) `TO.add` readValue c
  (y :: Tensor Value Float) <- TO.placeholder [ySize]
  let square_deltas = square (polynomial2_model `TO.sub` y)
  let loss = TO.reduceSum square_deltas
  trainStep <- TM.minimizeWith (TM.gradientDescent 0.0001) loss [a, b, c]
  let trainWithFeeds = \xF yF -> runWithFeeds
          [ feed x xF
          , feed y yF
          ]
          trainStep
  replicateM_ 10000
      (trainWithFeeds (encodeTensorData [xSize] xInput) (encodeTensorData [ySize] yInput))
  (Scalar a_learned, Scalar b_learned, Scalar c_learned) <- run (readValue a, readValue b, readValue c)
  return (a_learned, b_learned, c_learned)

someFunc :: IO ()
someFunc = do
  (a1, b1) <- learnParametersPolynomial1 [0, 1, 2, 3, 4] [1, 3, 5, 7, 9]
  _ <- putStrLn $ show a1 ++ "x + " ++ show b1
  (a2, b2, c2) <- learnParametersPolynomial2 [-2, -1, 0, 1, 2, 3, 4, 5] [9, 2, 1, 6, 17, 34, 57, 86]
  putStrLn $ show a2 ++ "x^2 + " ++ show b2 ++ "x + " ++ show c2

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Lib
    ( someFunc
    ) where

import Data.Int
import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TO hiding (initializedVariable)
import qualified TensorFlow.Variable as TF

someFunc :: IO ()
someFunc = do 
    (x :: Int64, y :: Int64) <- TF.runSession $ do
      let x = TO.vector [1 :: Int64, 2, 3, 4]
      w <- TF.initializedVariable 0
      b <- TF.initializedVariable 0
      (TF.Scalar w', TF.Scalar b') <- TF.run (TF.readValue w, TF.readValue b)
      return (w', b')
    putStrLn $ "X: " ++ show x ++ ", Y: " ++ show y

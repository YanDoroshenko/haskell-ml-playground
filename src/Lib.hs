module Lib
    ( someFunc
    ) where

import Control.Monad

someFunc :: IO ()
someFunc = forever $ putStrLn "someFunc"

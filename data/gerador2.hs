import System.IO
import Data.Char
import System.Directory
import System.Random
import Control.Monad (replicateM)
import System.TimeIt
import System.Timeout

putComma :: String -> String
putComma [] = []
putComma (x:[]) = [x]
putComma (x:xs) = x:[','] ++ putComma xs

removeComma :: String -> String
removeComma x = filter (/=',') x

tst = ["1010", "0010", "1110", "1111", "0001", "1"]

main :: IO()
main = do
    -- putStrLn "Start"
    -- timeIt $ putStrLn ("Result: " ++ show primesSum)
    -- putStrLn "End"
    exedir <- getCurrentDirectory
    contents <- readFile (exedir ++ "/4x4ruidorascunho2.csv")
    let possibilidades = drop 1 $ map (reverse . tail . tail . reverse) $ lines contents
    li <- replicateM 100000 $ replicateM (read $ sizeNumbers contents) ((randomRIO (0, 1) ) :: IO Int)
    -- putStrLn $ putComma $ foldl1 (++) $ map show $ head li
    --putStrLn $ controlData possibilidades $ map putComma $ flatToStr li 
    -- putStrLn $ controlData contents
    -- putStr $ unlines possibilidades
    -- putStrLn $ unlines $ clearTooMany $ flatToStr li
    writeFile (exedir ++ "/4x4ruidorandom2.csv") $ ( contents ) ++ ( controlData possibilidades $ map putComma $ flatToStr li )

sizeNumbers :: String -> String
sizeNumbers a = show $ div (length $ last $ lines a) 2

controlData :: [String] -> [String] -> String
controlData possibilidades aleatorios = unlines $ map (putComma . add0) $ clearDups ( remDups possibilidades ) $ map removeComma aleatorios

clearDups :: [String] -> [String] -> [String]
clearDups pos [] = []
clearDups pos (a:as)
    | elem a pos = as
    | otherwise = a : ( clearDups pos as )

clearTooMany :: [String] -> [String]
clearTooMany [] = []
clearTooMany (a:as)
    | ( sumFlatStr a ) > 8 || ( sumFlatStr a ) < 5 = clearTooMany as
    | otherwise = a : clearTooMany as

flatToStr :: [[Int]] -> [String]
flatToStr [] = []
flatToStr (x:xs) = ( foldl1 (++) $ map show x ) : flatToStr xs 

sumFlatStr :: String -> Int
sumFlatStr [] = 0
sumFlatStr (a:as) = read [a] + sumFlatStr as

add0 :: String -> String
add0 a = a ++ ['0']

remDups :: [String] -> [String]
remDups [] = []
remDups (x:xs)
    | elem x xs = remDups xs
    | otherwise = x : remDups xs 


import System.Directory

sumFlatStr :: String -> Int
sumFlatStr [] = 0
sumFlatStr (a:[]) = 0
sumFlatStr (a:as) = read [a] + sumFlatStr as

clearUnnecessary :: Int -> Int -> [String] -> [String]
clearUnnecessary a b [] = []
clearUnnecessary a b (x:xs)
    | ( sumFlatStr x ) > b || ( sumFlatStr x ) < a = clearUnnecessary a b xs
    -- | ( sumFlatStr a ) /= 5 = clearUnnecessary as
    | otherwise = x : clearUnnecessary a b xs

putComma :: String -> String
putComma [] = []
putComma (x:[]) = [x]
putComma (x:xs) = x:[','] ++ putComma xs

removeComma :: String -> String
removeComma x = filter (/=',') x

main :: IO()
main = do
    exedir <- getCurrentDirectory
    contents <- readFile (exedir ++ "/data1.csv")
    let fileWithNoCommas = removeComma contents
    let someDotsRemoved = clearUnnecessary 5 9 $ lines fileWithNoCommas
    writeFile (exedir ++ "/teste1.csv") $ unlines $ map putComma someDotsRemoved

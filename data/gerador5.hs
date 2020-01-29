import System.Directory

sumFlatStr :: String -> Int
sumFlatStr [] = 0
sumFlatStr (a:as) = read [a] + sumFlatStr as

clearUnnecessary :: [String] -> [String]
clearUnnecessary [] = []
clearUnnecessary (a:as)
    | ( sumFlatStr a ) > 9 || ( sumFlatStr a ) < 5 = clearUnnecessary as
    | otherwise = a : clearUnnecessary as

putComma :: String -> String
putComma [] = []
putComma (x:[]) = [x]
putComma (x:xs) = x:[','] ++ putComma xs

removeComma :: String -> String
removeComma x = filter (/=',') x

countClass :: [ String ] -> ( Int, Int ) -> ( Int, Int )
countClass [] ( um, zero ) = ( um, zero )
countClass ( a:as ) ( um, zero )
    | last a == '1' = countClass as ( um+1, zero )
    | otherwise = countClass as ( um, zero+1 )

main :: IO()
main = do
    exedir <- getCurrentDirectory
    contents <- readFile (exedir ++ "/teste1.csv")
    let count1 = fst $ countClass ( lines contents ) ( 0, 0 )
    let count0 = snd $ countClass ( lines contents ) ( 0, 0 )
--    putStrLn $ "1: " ++ ( show count1 ) ++ " | 0: " ++ ( show count0 )
    let class1 = filter ( ( == '1' ).last ) $ lines contents
    let class0 = filter ( ( == '0' ).last ) $ lines contents
    let class2 = filter ( ( == '2' ).last ) $ lines contents
    writeFile (exedir ++ "/4noises.csv") $  unlines $ class1 ++ class0 ++ class2


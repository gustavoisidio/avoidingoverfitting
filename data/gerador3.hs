import System.Directory

plus1 :: String -> String
plus1 [] = "1"
plus1 ( a:as ) 
    | a == '0' = ( '1' : as )
    | otherwise = ( '0' : ( plus1 as ) )

runN :: Int -> String -> String
runN 0 s = s
runN n s = runN ( n-1 ) ( plus1 s )

generateall :: String -> [ String ] -> [ String ]
generateall end [] = generateall end [ plus1 "" ]
generateall end ( a:as )
    | a == end = a:as 
    | otherwise = generateall end $ ( plus1 a ) : ( a:as )

putComma :: String -> String
putComma [] = []
putComma (x:[]) = [x]
putComma (x:xs) = x:[','] ++ putComma xs

giveme0 :: Int -> String
giveme0 0 = []
giveme0 n = "0" ++ giveme0 ( n-1 )

completeAll :: Int -> [ String ] -> [ String ]
completeAll size [ ] = []
completeAll size ( a:as ) = ( with0 size a ) : completeAll size as
    where with0 s x = ( giveme0 ( s - ( length x ) ) ) ++ x

sumFlatStr :: String -> Int
sumFlatStr [] = 0
sumFlatStr (a:[]) = 0
sumFlatStr (a:as) = read [a] + sumFlatStr as

clearUnnecessary :: [String] -> [String]
clearUnnecessary [] = []
clearUnnecessary (a:as)
    | ( sumFlatStr a ) > 6 || ( sumFlatStr a ) < 5 = clearUnnecessary as
    | otherwise = a : clearUnnecessary as

main :: IO()
main = do
    let max = "1111111111111111"
    let allPossibilities = generateall max [  ]
    let possibilitiesInCorrectOrientation = map reverse $ reverse $ allPossibilities
    let possibilitiesInCorrectSize = completeAll ( length max ) $ possibilitiesInCorrectOrientation
    let possibilitiesBetween5and8 = clearUnnecessary possibilitiesInCorrectSize
    let allPossibilitiesWithComma = map putComma possibilitiesBetween5and8
    writeFile ("allgenerated.csv") $ unlines allPossibilitiesWithComma 



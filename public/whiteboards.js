// function removeCharsFromEnd(str, chars_to_remove) {
//     for (let pattern of chars_to_remove) {
//         if (str.endsWith(pattern)) {
//             return str.slice(0, -pattern.length);
//         }
//     }
//     return str; // Return the original string if no pattern matches
// }

// let str = "example_S_V";
// let chars_to_remove = ['_S_V', '_S', '_V'];

// let result = removeCharsFromEnd(str, chars_to_remove);
// console.log(result); // Outputs: "example"



// function highAndLow(numbers){
//     // can split at spaces to get each number
//     // 
//     let integers = numbers.split(' ').map(Number)
//     return `${Math.max(...integers)} ${Math.min(...integers)}`
// }

// console.log(highAndLow("1 2 -3 4 5"));


// function* generator(a) {
//     let b = 1
//     while (true) {
//         yield `${a} * ${b} = ${a*b}`
//         b += 1
//     }
// }

// const iterator = generator(4);

// console.log(iterator.next()); // { value: 1, done: false }
// console.log(iterator.next()); // { value: 2, done: false }
// console.log(iterator.next()); // { value: 3, done: false }
// console.log(iterator.next()); // { value: undefined, done: true }  


// var isSquare = function(n){
//     return n ** (1/2)
//   }

// console.log(isSquare(16))


// // count the occurrences of each char in a string

// // {s:3, u:1, etc}

// // object variable required
// // map key value pairs

// const countingCharacters = (str) => {

//     let charCount = {}
//     let maxCount = 0
//     let mostOccuring = {}

//     for (let i=0; i < str.length; i++) {

//         let char = str[i]

//         if (char === ' ') {
//             continue 
//         }
//         if (!charCount[char]) {
//             charCount[char] = 1
//         } else {
//             charCount[char]++
//         }

//         if (charCount[char] > maxCount) {
//             maxCount = charCount[char]
//             mostOccuring = {[char]: char} 
//         }
//     }

//     return charCount 
// }

// console.log(countingCharacters('tomorrow'));



// function solve(arr){  
//     const alph = 'abcdefghijklmnopqrstuvwxyz'

//     numArr = []
//     for (let word of arr){
//         counter = 0
//         for (let i in word){
//             word[i].toLowerCase() === alph[i] ? counter += 1 : null 
//         }
//         numArr.push(counter)
//     }
//     return numArr
// };

// console.log(solve(["abode","ABc","xyzD"]));

// function perimeterSequence(a,n) {
  
//   return
// }

// console.log(perimeterSequence(1,3));


// function vowelOne(s){
//     // string with vowels, for loop to check each char, if in vowels, push '1' to list, else push '0'
//     let final = ''
//     const vowels = 'aeiouAEIOU'
//     for (c of s){
//         vowels.includes(c) ? final += '1' : final += '0'
//     }
//     return final 
//   }

// console.log(vowelOne("aeiou, abc")); // 1111100100


// function isSortedAndHow(array) {
//     let prev = array[0]
//     for (let n of array){
//         if (n >= prev){
//             prev = n 
//         } else break 
//     }
//     return 'sorted asc'
// }

// console.log(isSortedAndHow([1,2,100,]));


// function isLeapYear(year) {
//     // years div by 4 ARE leap yrs EXCEPT IF also div by 100
//     // div by 100 ARE NOT leap years EXCEPT IT yrs div by 400
//     // conc: all years div by 4 are leap yrs EXCEPT if div by 100 EXCEPT IF div by 400
//     // if yr div by 400, LEAP YR
//     // else if yr div by 100, NOT LEAP YR
//     // else if yr dib by 4, LEAP YR
//     // else NOT LEAP YR
//     return year % 400 === 0 ? true : year % 100 === 0 ? false : year % 4 === 0 ? true : false 
//   }

// console.log(isLeapYear(504));


// function sumCubes(n){
//   sum = 0
//   for (let i=1; i<n+1; i++){
//     sum += i**3
//   }
//   return sum 
// }

// console.log(sumCubes(3));

// let array = new Array(20).fill(0);

// console.log(array);



// var greet = function(name) {
//   return 'Hello ' + name.slice(0,1).toUpperCase() + name.slice(1).toLowerCase() + '!'
// };

// console.log(greet('JACK'));


// function partsSums(lst) {
//   let final = []
//   for (i in lst){
//     final.push(lst.slice(i).reduce((a,b)=>a+b))
//   }
//   final.push(0)
//   return final
// }

// console.log(partsSums([0,1,3,6,10]));


// function nicknameGenerator(name){
//     const vowels = 'aieou'
//     return name.length < 4 ? 'Error: Name too short' : vowels.includes(name[2]) ? name.slice(0,4) : name.slice(0,3)
//   }

// console.log(nicknameGenerator("Samantha"));
// console.log(nicknameGenerator("Alex"));


// function dataReverse(data) {
//     // need to split array into multiple arrays, then just reverse the orders of the arrays, not their contents
//     let final = []
//     let eightbits = []
//     let counter = 0
//     for (let i=0; i<data.length+1; i++){
//         if (counter === 8) {
//             final.push(eightbits)
//             eightbits = []
//             counter = 0
//         }
//         eightbits.push(data[i])
//         counter += 1
//     }
//     return final.reverse().flat()
//   }

// console.log(dataReverse([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,0,1,0,1,0]));


// function pattern(n){
//     let output = ""
//     for (i=1; i < n+1; i++){
//         i === 1 ? output += `${i}` : output += "\\n" + i.toString()*i
//     } 
//     return output 
//    }

// console.log(pattern(5));


// function alternateCase(s) {
//     let alternate = [...s].map(c => c.toUpperCase() === c ? c.toLowerCase() : c.toLowerCase() === c ? c.toUpperCase() : c)
//     return alternate.join('')
//   }

// console.log(alternateCase('Hello World'));


// function balancedNum(number) {
//     // need to split string into two sides, left and right. 
//     // if string even, ignore middle 2 nums, if odd, ignore middle num
//     let init = number.toString().length
//     len = init % 2 === 0 ? init/2-1 : Math.floor(init/2) 
//     if (init < 3){return 'Balanced'}
//     if (
//         [...number.toString().slice(0,len)].reduce((a,b)=>parseInt(a)+parseInt(b)) 
//         ===
//         [...number.toString().slice(-3)].reduce((a,b)=>parseInt(a)+parseInt(b))
//     ) {
//         return "Balanced" 
//     }
//     return "Not Balanced"
// }

// console.log(balancedNum(562398143)); // balanced


// function growingPlant(upSpeed, downSpeed, desiredHeight) {
//     let curr_height = 0
//     let day = 0
//     while (curr_height < desiredHeight){
//         curr_height += upSpeed
//         day += 1
//         if (curr_height >= desiredHeight){return day}
//         curr_height -= downSpeed
//     }
//   }

// console.log(growingPlant(10,9,4));


// var flatten = function (array){
//   return array.flat(3)
// }

// console.log(flatten([[1,2,3],["a","b","c"],[1,2,3]]));


// function countRedBeads(n) {
//   // start 1 blue bead. for every addtl blue bead, there's 2 red beads
//   return n < 2 ? 0 : (n-1) * 2
// }

// console.log(countRedBeads(5));


// function killer(suspectInfo, dead) {
//     for (suspect of suspectInfo) {
//         console.log(suspectInfo[suspect]);
//     }
//   }

// console.log(killer({'James': ['Jacob', 'Bill', 'Lucas'], 'Johnny': ['David', 'Kyle', 'Lucas'], 'Peter': ['Lucy', 'Kyle']}, ['Lucas', 'Bill']));


// function getSumOfDigits(integer) {
//     var sum = 0;
//     var digits =  integer.toString();
//     for (var ix = 0; ix < digits.length; ix++) {
//       sum += Number(digits[ix]);
//     }
//     return sum;
//   }

// console.log(getSumOfDigits(123));


// const orderedCount = function (text) {
//     let final = []
//     let seen = []
//     for (c of text){
//         if (!seen.includes(c)) {
//             final.push([c, [...text].filter(char => char == c).length])
//             seen.push(c)
//         }
//     }
//     return final 
//   }

// console.log(orderedCount('abracadabra'));


// function solution(start, finish) {
//     // can only jump either to next shelf, or + 3 shelfs, not + 2
//     return Math.floor((finish - start) / 3) + (finish - start) % 3
// }
  
//   // Example usage
//   console.log(solution(916,1400)); // Output: 2
  


// var filterString = function(value) {
//   return Number(value.split('').filter(n=>Number(n)).join(''))
// }

// console.log(filterString("aa1bb2cc3dd"));


// function cookingTime(eggs) {
//     // 8 is the max num of eggs in the only pot. so 5mins limit for 8 eggs.
//     return Math.ceil(eggs/8) * 5
//   }

// console.log(cookingTime(17));


// function sumOfMinimums(arr) {
//     return arr.map(inArr => (Math.min(...inArr))).reduce((a, b) =>{console.log(a,b); return a + b}, 0)
// }

// console.log(sumOfMinimums([[7, 9, 8, 6, 2], [6, 3, 5, 4, 3], [5, 8, 7, 4, 5]]));


// function minSum(arr) {
//     arr.sort((a, b) => a - b); // Sort the array in ascending order
  
//     let minSum = 0;
//     for (let i = 0; i < arr.length / 2; i++) {
//       minSum += arr[i] * arr[arr.length - 1 - i];
//     }
  
//     return minSum;
//   }
  
//   // Example usage:
//   const arr = [5, 4, 2, 3];
//   const result = minSum(arr);
//   console.log(result); // Output: 22
  


// function minSum(arr) {
//     arr = arr.sort(function (a, b) {return a - b});
//     let i = 0;
//     let j = arr.length - 1;
//     let sum = 0;
//     while (i < j ) {
//       sum += arr[i] * arr[j];
//       i++;
//       j--;
//     }
//     return sum;
//   }

// console.log(minSum([5,4,2,3]));


// function sortMyString(S) {
//     let even = ''
//     let odd = ''
//     for (let i in S){
//         i % 2 !== 0 ? odd += S[i] : even += S[i]
//     }
//     return even + ' ' + odd 
// }

// console.log(sortMyString('CodeWars'));


// function squareDigits(num){
//     return parseInt(num.toString().split('').map(a=>(a**2).toString()).join(''))
//   }

// console.log(squareDigits(9212));


// function countDevelopers(list) {
//     let counter = 0
//     list.forEach(dev => dev.language == 'JavaScript' && dev.continent == 'Europe' ? counter += 1 : 0)
//     // for (let dev of list){
//     //     dev.language == 'JavaScript' && dev.continent == 'Europe' ? counter += 1 : 0
//     // }
//     return counter 
//   }

// console.log(countDevelopers([
//     { firstName: 'Noah', lastName: 'M.', country: 'Switzerland', continent: 'Europe', age: 19, language: 'JavaScript' },
//     { firstName: 'Maia', lastName: 'S.', country: 'Tahiti', continent: 'Oceania', age: 28, language: 'JavaScript' },
//     { firstName: 'Shufen', lastName: 'L.', country: 'Taiwan', continent: 'Asia', age: 35, language: 'HTML' },
//     { firstName: 'Sumayah', lastName: 'M.', country: 'Tajikistan', continent: 'Asia', age: 30, language: 'CSS' }
//   ]));


// function evenNumbers(array, number) {
//     return array.filter(a => a % 2 == 0).slice(-number)
//   }

// console.log(evenNumbers([-22, 5, 3, 11, 26, -6, -7, -8, -9, -8, 26], 2));


// function getCount(str) {
//     let counter = 0
//     const vowels = 'aeiou'
//     for (c of str){
//         vowels.includes(c) ? counter += 1 : null 
//     }
//     return counter 
//   }

// console.log(getCount("abracadabra"))

// function disemvowel(str) {
//     // // advanced
//     // const vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
//     // return str
//     //   .split('')
//     //   .filter(char => !vowels.includes(char))
//     //   .join('')
    
//     // basic
//     let vowels = 'aeiouAEIOU'
//     let final = ''
//     for (c of str){
//         !vowels.includes(c) ? final += c : null 
//     }
//     return final 
//   }

// console.log(disemvowel("This website is for losers LOL!"))


// function largest(n, array) {
//     // Find the n highest elements in a list
//     return array.sort((a,b)=>b-a).slice(0, n).sort((a,b)=>a-b)
//   }

// console.log(largest(2, [5,6,7,4,3,2,100]));


// function solve(arr) {
//     let unique = []
//     for (i=arr.length-1; i>=0; i--){
//         unique.includes(arr[i]) ? null : unique.push(arr[i])
//     }
//     return unique.reverse()
//   }

  
//   console.log(solve([3,4,4,3,6,3]));


// function toNumberArray(stringarray){
//   return stringarray.map(a => Number(a))
// }

// console.log(toNumberArray(["1","2.2","3.3"]));

// let multiline = `
//   This is a\
//   multiline
//   string.
// `;
// console.log(multiline);


// function findLongest(array){
//     // need to count num of digits in each number in array
//     let largest = array[0]
//     for (let n of array){
//         n.toString().length > largest.toString().length ? largest = n : null
//         // console.log(n.toString().length)
//         // console.log(n)
//     }
//     return largest 
//   }

// console.log(findLongest([8, 900, 500]));


// function solution(digits){
// //   for loop that takes in current iter digit and 4 following digits, update a var if number > var
//     let max = 0
//     for (i=0; i <= digits.length-4; i++){
//         let num = parseInt(digits.substring(i, i+5))
//         if (num > max){
//             max = num 
//         }
//     }
//     return max 
// }

// console.log(solution('731674765'));

// function solution(number) {
//     let greatestSequence = 0;
    
//     for (let i = 0; i < number.length - 4; i++) {
//       const sequence = parseInt(number.substring(i, i + 5));
//       if (sequence > greatestSequence) {
//         greatestSequence = sequence;
//       }
//     }
    
//     return greatestSequence;
//   }
  



// function alphabetWar(fight){
//     let left = {'w': 4, 'p': 3, 'b': 2, 's': 1}
//     let right = {'m': 4, 'q': 3, 'd': 2, 'z': 1}
//     let lt = 0
//     let rt = 0
//     for (let c of fight){
//         c in left ? lt += left.c : c in right ? rt += right.c : null 
//         console.log(rt, lt )
//     } 
//     return lt, rt 
//     // return lt > rt ? 'Left side wins!' : rt > lt ? 'Right side wins!' : `Let's fight again`
//     // return "Let's fight again!";
// }

// console.log(alphabetWar('zdqmwpbs'));


// function sumTriangularNumbers(n) {
//     // need to sum ...
//     let count = 0
//     let final = 0
//     for (let i=1; i<n+1; i++){
//         count += i
//         final += count 
//     }
//     return final 
//   }

// console.log(sumTriangularNumbers(6));


// const bump = x => x.split("").filter(char => char === 'n').length > 15 ? 'Car Dead' : 'Woohoo!'

// console.log(bump("_nnnnnnn_n__n______nn__nn_nn"));


// // return the two oldest/oldest ages within the array of ages passed in.
// function twoOldestAges(ages){
//   return ages.sort((a,b) => a-b).slice(-2)
// }

// console.log(twoOldestAges([1, 5, 87, 45, 8, 8]));

// function vowelIndices(word) {
//     const vowels = ['a', 'e', 'i', 'o', 'u'];
//     const indices = [];
//     for (let i = 0; i < word.length; i++) {
//       if (vowels.includes(word[i])) {
//         indices.push(i + 1);
//       }
//     }
//     return indices;
//   }
  

// console.log(vowelIndices('apple'));

// function divCon(x){
//     let plus = 0
//     let minus = 0
//     for (num of x){
//         console.log(typeof num);
//     }
//     return plus
// }

// console.log(divCon([9, 3, '7', '3']));


// function add(n) {
//   return function addAlso(m) {
//     return n+m;
//   }
//   // function also(m){
//   //   return n + m
//   // }
//   // return also 
// }

// console.log(add(5)(2));


// // function solve(s){
// //     // upper: 65-90
// //     // lower: 97-122
// //     // numbers: 30-39
// //     // special: all others
// //     let upper = 0
// //     let lower = 0
// //     let number = 0
// //     let special = 0
// //     for (let i in s){
// //         let x = s.charCodeAt(i)
// //         x>=65 && x<=90 ? upper += 1 : x>=97 && x<=122 ? lower += 1 : x>=49 && x<=58 ? number += 1 : special += 1
// //     }
// //     return [upper, lower, number, special]
// // }

// const solve = x => {
//     let u = x.match(/[A-Za-z0-9*]/g)||[]
//     let d = x.match(/[a-z]/g)||[]
//     let n = x.match(/[0-9]/g)||[]
//     let s = x.match(/[^A-Z0-9]/gi)||[]
//     return [u, d.length, n.length, s.length]
//   }

// console.log(solve("*'&ABCDabcde12345"));


// function digitize(n) {
//   return [...n.toString()].map(Number)
// }

// console.log(digitize(8675309));


// function sortGiftCode(code){
//   return code.split('').sort().join('')
// }

// console.log(sortGiftCode('zyxwvutsrqponmlkjihgfedcba'));

// function removeDuplicateWords(s){
//     // your perfect code...
//     // to remove words, need to separate by white spaces first into a list outside for loop
//     // in for loop, check if current word in 
//     return [...(new Set(s.split(" ")))].join(" ")
//   }

// console.log(removeDuplicateWords("alpha beta beta gamma gamma gamma delta alpha beta beta gamma gamma gamma delta"))


// function sumDigits(number){
//     return (Math.abs(number).toString().split('')).map(c => Number(c)).reduce((a, b) => a + b)
//     // return (Math.abs(number).toString().split('')).reduce((a, b) => +a + +b)
// }

// console.log(sumDigits(-99));

// function oddOrEven(array) {
//     let sum = 0
//     for (n of array){
//         sum += n 
//     }
//     return sum % 2 === 0 ? 'even' : 'odd'
//  }

//  console.log(oddOrEven([0, -1, 5]));

// Make a program that filters a list of strings and returns a list with only your friends name in it.

// If a name has exactly 4 letters in it, you can be sure that it has to be a friend of yours! 
// Otherwise, you can be sure he's not...

// Ex: Input = ["Ryan", "Kieran", "Jason", "Yous"], Output = ["Ryan", "Yous"]

// i.e.

// friend = ["Ryan", "Kieran", "Mark"] `shouldBe` ["Ryan", "Mark"]


// let input = ["Ryan", "Kieran", "Jason", "Yous"]

// function friend(names){
//     return names.filter(element => element.length === 4);
// };

// console.log(friend(input));



// // 2
// // need to check each iterable of string, check if it is capital letter, push capital letters to new array
// function ord_str(word){
//     let capitals = []
//     for (let index in word){
//         if (word[index] == word[index].toUpperCase()){
//             capitals.push(Number(index))
//         }
//     }
//     return capitals
// };

// console.log(ord_str('CodEWaRs'));
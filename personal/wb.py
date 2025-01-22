# INCLUDE THIS ALWAYS!!!
from typing import Optional, List
# INCLUDE THIS ALWAYS!!!


class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        greatest = max(candies)
        for i in range(len(candies)):
            if candies[i] + extraCandies >= greatest:
                candies[i] = True
            else:
                candies[i] = False
        return candies 
    
print(Solution().kidsWithCandies([2,3,5,1,3], 3))



# # 1071. Greatest Common Divisor of Strings
# class Solution:
#     def gcdOfStrings(self, str1: str, str2: str) -> str:
#         # if you split at 1+ chars present in both strings, should be empty list in both, that is the divisor
#         for i in range(min(len(str1), len(str2)), -1, -1):
#             try:
#                 if set(str1.split(str1[:i])) == set(str2.split(str1[:i])) == {''}:
#                     return str1[:i]
#             except ValueError:
#                 return ''
        


# print(Solution().gcdOfStrings('abcabc', 'abc'))



# class Solution:
#     def mergeAlternately(self, word1: str, word2: str) -> str:
#         # so just for each letter in word (until reaching end of shortest word) add alternately to final string
#         # final = ''
#         # longest = 1 if len(word1) > len(word2) else 2
#         # last_i = 0
#         # for i in range(0, min(len(word1), len(word2))):
#         #     final += word1[i] + word2[i]
#         #     last_i = i + 1
#         # return final + word1[last_i:] if longest == 1 else final + word2[last_i:]
#         final = ''
#         for i in range(0, max(len(word1), len(word2))):
#             if i < len(word1):
#                 final += word1[i]
#             if i < len(word2):
#                 final += word2[i]
#         return final 


# print(Solution().mergeAlternately('abcd', 'pq'))


# class Solution:
#     def majorityElement(self, nums: List[int]) -> int:
#         # majority element is the element that appears more than n/2 times
#         # most logical is maintain a count of the elements seen in for loop
#         count_dict = {}
#         # perhaps something like storing just one counter, updating
#         # what if you divide the list in half, check majority in both?

#         for n in nums:
#             count_dict[n] = count_dict.get(n, 0) + 1
#             if count_dict[n] > len(nums) // 2:
#                 return n 
            

    
    
# print(Solution().majorityElement([2,2,1,1,1,2,2]))



# class Solution:
#     def convertToTitle(self, n: int) -> str:
#         # given integer, return excel column title
#         # first 26 straightfwd
#         # pattern every 26 is to add a new letter to the left
#         # A, AA, AAA, AAAA, AAAAA
#         # B, AB, AAB, AAAB, AAAAB
#         # should be doable with just one str ref for alphabet
#         alph = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#         # return self.convertToTitle(n // 26) + alph[n % 26 - 1] if n > 26 else alph[n-1]
#         if n < 26:
#             return alph[n-1]
#         return self.convertToTitle((n // 26+1)-1) + alph[n % 26 -1]


# base = 678
# print(Solution().convertToTitle(base))
# print(base//26, base%26)

# # 160. Intersection of Two Linked Lists
# # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# class Solution:
#     def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
#         # O(1) memory
#         # if intersection node, that node will have 2 pointers to it, so its prev nodes will point to it and are the key (but have to keep the list intact in same order)
#         # or the key could be in getting the list length...if the lists are different lengths, then the intersection node will be the same distance from the end of the list
#         # if diff lengths, start the longer one at the same distance from the end as the shorter one, then move both by 1 until they meet
#         len_a, len_b = 0, 0
#         curr_a, curr_b = headA, headB
#         while curr_a or curr_b:
#             if curr_a:
#                 len_a += 1
#                 curr_a = curr_a.next 
#             if curr_b:
#                 len_b += 1
#                 curr_b = curr_b.next 
#         curr_a, curr_b = headA, headB
#         if len_a > len_b:
#             # set curr to be same len as other list
#             while len_a > len_b:
#                 curr_a = curr_a.next
#                 len_a -= 1
#         elif len_b > len_a:
#             # set curr to be same len as other list
#             while len_b > len_a:
#                 curr_b = curr_b.next
#                 len_b -= 1
#         # else means they're equal, so start final while loop
#         while curr_a and curr_b:
#             if curr_a == curr_b:
#                 return curr_a
#             curr_a, curr_b = curr_a.next, curr_b.next 
#         return None 
        
        
#         # # best way i can think to solve is to store the seen nodes in a list, and check for each list if node in seen
#         # seen = set()
#         # # while not seen node and while either or both lists still have next node, check the active list's next node if in seen
#         # while headA:
#         #     seen.add(headA)
#         #     headA = headA.next
#         # while headB:
#         #     if headB in seen:
#         #         return headB 
#         #     headB = headB.next 
#         # return None 

# headA = ListNode(1)
# headA.next = ListNode(3)
# headB = ListNode(2)
# headB.next = headA.next 

# print(Solution().getIntersectionNode(headA, headB))



# # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
#         # post is leftmost leaf first, then right, then move upwards
#         final = []

#         def dfs(root):
#             if not root:
#                 return 
#             # first visit left, then right 
#             dfs(root.left)
#             dfs(root.right)
#             final.append(root.val)

#         dfs(root)
#         return final 


# root = TreeNode(1)
# root.left = TreeNode(2)
# root.left.left = TreeNode(4)
# root.left.right = TreeNode(5)
# root.right = TreeNode(8)


# print(Solution().postorderTraversal(root))


# # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
#         # you visit first left then right..should return order top >> down, left >> right
#         if not root:
#             return [] 
#         return [root.val] + \
#             self.preorderTraversal(root.left) + \
#             self.preorderTraversal(root.right)
    

# root = TreeNode(1)
# root.left = TreeNode(2)
# root.left.left = TreeNode(4)
# root.left.right = TreeNode(5)
# root.right = TreeNode(8)


# print(Solution().preorderTraversal(root))


# def build_palindrome(s):
#     pass
#     # just need to check if front or back limited numbers of chars will be added to str
#     # left, right vars
#     # keep adding to left from right and vice versa chars and check each time if palindrome success, once success switch from left to right
#     left, right = s, s 
#     attempts = 1
#     ladd, radd = -1, 0
#     while left != left[::-1] and len(s)//attempts > 1:
#         left = left[ladd] + left
#         print(left)
#         attempts += 1
#         ladd -= 1
#     attempts = 1
#     while right != right[::-1] and len(s)//attempts > 1:
#         right = right + right[radd]
#         print(right)
#         attempts += 1
#         radd += 1
#     return left if len(left) <= len(right) else right 
    

# print(build_palindrome('right'))


# # 142. Linked List Cycle II
# # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# class Solution:
#     def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
#         # do not create doubly linked list
#         # two pointer sol that somehow catches the being value...
#         # how to check if a node has two pointers to it? reverse it, and check if still has next value (means start?)
#         fast, slow = head, head 
#         while fast and fast.next:
#             # means there is a cycle, continue w/code 
#             slow = slow.next 
#             fast = fast.next.next
#             if fast == slow:
#                 break 
#         else:
#             # no cycle found
#             return None 
        
#         # now reset the slow pointer to head, keep fast, move both by 1 until meeting
#         slow = head 
#         while fast != slow:
#             fast = fast.next 
#             slow = slow.next 
#         return fast 
        
#         # # o(n) time, o(n) space first
#         # # somehow store the index of the node being checked 
#         # # could store all nodes, check if new next node already seen..
#         # curr = head 
#         # seen = set()
#         # while curr:
#         #     if curr in seen:
#         #         return curr
#         #     else:
#         #         seen.add(curr)
#         #         curr.next
#         # return None 


# l1 = ListNode(10)
# l1.next = ListNode(20)
# l1.next.next = l1
# print(Solution().detectCycle(l1))


# # 141. Linked List Cycle
# # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# class Solution:
#     def hasCycle(self, head: Optional[ListNode]) -> bool:  # pos
#         # the tail node will either be the last with none next, or not..
#         # need pos? yes, but how to access?
#         # did not require pos, just distraction
#         # two pointers, one fast, one slow will eventually meet always if cycle
#         slow = head
#         fast = head

#         while fast and fast.next:
#             fast = fast.next.next
#             slow = slow.next
#             # if not fast or fast = slow, return true
#             if fast == slow:
#                 return True

#         return False



# # 137. Single Number II
# class Solution:
#     def singleNumber(self, nums: List[int]) -> int:
#         # XOR cancels repeating nums out. but how to get rid of 3rd? could match to other prev matches?
#         # constant space so can't store linear items
#         # somehow would cancel all duplicates first, then get sum of list, subtract sum of duplicates from sum of list to get unique num
#         # would prob require 2 constant vars
#         res = 0
#         sum_all = sum(nums)
#         return 


# print(Solution().singleNumber([0,1,0,1,0,1,99]))



# # 136. Single Number
# class Solution:
#     def singleNumber(self, nums: List[int]) -> int:
#         # O(n) time, O(1) space
#         # O(1) space means you cannot store more than 1 var at a time?
#         # you do need to iterate through entire list, no way around it
#         # if you make all duplicates one positive one negative, would get zero, except for the unique value...

#         seen = set()
#         for n in nums:
#             if n in seen:
#                 seen.remove(n)
#             else:
#                 seen.add(n)
#         return seen.pop()

# print(Solution().singleNumber([4,1,2,1,2]))


# # 127. Word Ladder
# class Solution:
#     def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
#         pass 
#         # need to change just one letter from each consecutive word and get to the final word
#         # all word lengths equal
#         # think you could just find words that replace a single char in curr word with char in final word?
#         # meme >>> team. meme > mome > tome > teme > teae > team
#         #   you could start right at teme since that's one char dif, but there are essentially multiple ways to solve...so how to find the fastest way to solve? would need to go through all possibilities? yes, unless you find a possibility that's equal to num of diff words bw start and end word. so meme to team requires at least 3 transforms. meme,teme,teae,team. so if not len(diff in chars bw start, end words) == num transformations, keep seaching for most efficient solution
#         # could there be a case where you transform to a letter that is not in final word same index? song >>> beam. song > sang > bang > beng > beag > beam
#         # yes, you could use other letters to reach final...very hard problem



# # 125. Valid Palindrome
# class Solution:
#     def isPalindrome(self, s: str) -> bool:
#         # convert all to lowercase
#         # remove all non-alphanumeric chars
#         # stripped = tuple(filter(lambda c: c.isalnum(), s.lower()))
#         # l, r = 0, len(stripped)-1
#         # while l <= r: 
#         #     if stripped[l] != stripped[r]:
#         #         return False 
#         #     l, r = l+1, r-1
#         # return True 
    
#         s = ''.join(c for c in s.lower() if c.isalnum())
#         return s == s[::-1]


# print(Solution().isPalindrome('Race  car'))



# # 124. Binary Tree Maximum Path Sum
# # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def maxPathSum(self, root: Optional[TreeNode]) -> int:
#         # just a contiguous path that doesn't split up, continuous 
#         # could also return just one node if that's the max path
#         # recursive
#         # how do you prevent splitting subtrees in answer?
#         # if not main root, and if parent has 2 child nodes, cannot venture past higher than parent node w/both children if path is longer than child>parent>child, this would create invalid split path.
#         res = [root.val]  # this needs to be mutable obj to access its values within nested functions, otherwise can't reassign outer variable in nested fn

#         # return the max path sum w/o splitting
#         def dfs(root):
#             if not root:
#                 return 0 

#             left_max = dfs(root.left)
#             right_max = dfs(root.right)
#             left_max = max(left_max, 0)
#             right_max = max(right_max, 0) 

#             # compute max path sum WITH split
#             # update res[0], compare res[0] vs current subtree sum value
#             # check how this is NOT grabbing all split sum paths, just linear paths
#             res[0] = max(res[0], root.val + left_max + right_max)

#             return root.val + max(left_max, right_max)

#         dfs(root)
#         return res[0]


# root = TreeNode(-10)
# root.left = TreeNode(9)
# root.right = TreeNode(20)
# root.right.left = TreeNode(15)
# root.right.right = TreeNode(7)
# print(Solution().maxPathSum(root))        



# # 123. Best Time to Buy and Sell Stock III
# # You are given an array prices where prices[i] is the price of a given stock on the ith day.
# # Find the maximum profit you can achieve. You may complete at most two transactions.
# # Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
# # Example 1:
# # Input: prices = [3,3,5,0,0,3,1,4]
# # Output: 6
# # Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
# # Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         # what i would do personally, is check all trough to peaks (meaning lowest to highest) and get their values as possible profit values, then compare them with each other depending on their index values to see which pairs are both highest profit and compatible in range
#         # if holding stock (bought) you cannot buy another, must sell prior
#         # you can either never buy a stock, buy and sell once, or buy and sell twice
#         # if you buy and sell twice, cannot hold 2 at once, must sell prior
#         # cannot sell then buy on same day (should be no reason to)
#         # could be O(2n) problem, you just run through list twice to check best?
#         # keep track if holding stock, cannot buy while holding
#         # multiple possible options, so need to compare all options
#         # for each num, check if buy that num, what is max profit achievable in 1 or 2 buy/sell actions from that num forward
#         # no overlap so grab the two highest profits

    
# print(Solution().maxProfit([3,3,5,0,0,3,1,4]))
        

# # best time to buy and sell stock II
# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         # common sense: should only buy when very low compared to future values
#         # should only sell when high compared to future values..
#         # always need to check if holding 1 share or not, start w/o share
#         # pattern: check next value, only buy if NOT holding and next is lower
#         # only sell if holding and next is higher
#         # not enough, need to check consecutive values
#         # could run through list once, check peaks and troughs, add index of those to 2 lists, then in final check if index to buy or sell based on peaks and troughs
#         # you can buy and sell on same day. either buy and sell or sell and buy
#         profit = 0
#         holding = False
#         last_value = 0
#         prices += [0]
#         for i in range(len(prices)-1):
#             # always sell if you're holding no matter price
#             if holding:  # means prev price was lower so sell
#                 holding = False
#                 profit += prices[i] - last_value 
#             # if not holding, only buy if next price is higher
#             if prices[i+1] > prices[i]:
#                 last_value = prices[i]
#                 holding = True 
#         return profit

# print(Solution().maxProfit([7,5,1,3,6,10,1,200,1]))


# best time to buy and sell stock
# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         # grabbing min/max doesn't achieve anything since depends on placement
#         # could sort numbers, do a two pointers to check
#         # most obvious is n^2 solution go through each...
#         # is there o of n solution?
#         # 
#         if len(prices) < 2: return 0

#         max_profit = 0
#         curr_min = prices[0]

#         for i in range(1, len(prices)):
#             if prices[i] < curr_min:
#                 curr_min = prices[i]
#                 continue
#             if prices[i] - curr_min > max_profit:
#                 max_profit = prices[i] - curr_min 
                
#         return max_profit 


# print(Solution().maxProfit([99,100,4,6,1,2]))



# # Triangle
# class Solution:
#     def minimumTotal(self, triangle: List[List[int]]) -> int:
#         # len of triangle is the total num of rows
#         # so you can only move to the row below's same index or index + 1 and that's it
#         # how can you check min path on first glance? impossible need to iterate through all possibilities
#         # could try the inefficient route, all possible options
#         # not straightfwd algorithm. 
#         return


# print(Solution().minimumTotal([[2],[3,4],[6,5,7],[4,1,8,3]]))




# # Pascals triangle 2
# class Solution:
#     def getRow(self, rowIndex: int) -> List[int]:
#         if rowIndex == 0:
#             return [1]
#         elif rowIndex == 1:
#             return [1,1]
        
#         prev = [1,1]  # acct for this later
#         for i in range(rowIndex-1):
#             # pattern is add 1 list item to past list, add all consecutive pairs from past list, outer nums always equal 1
#             item = [1]  # always 1 at start of item
#             for i in range(len(prev)-1):
#                 item.append(prev[i] + prev[i+1])
#             item.append(1)  # always 1 at the end of item
#             prev = item

#         return item


# Pascals triangle
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        # should return list of numRows items, starting with [1], [1,1]
        # pattern is all outer nums will always be 1, what changes is nums within starting with 3rd item
        # each subsequent item has an additional num in its list, so 1,2,3,4,5,... items

        # base case
        if numRows == 1:
            return [[1]]
        elif numRows == 2:
            return [[1], [1,1]]
        
        final = [[1], [1,1]]  # acct for this later
        for i in range(numRows-2):
            # pattern is add 1 list item to past list, add all consecutive pairs from past list, outer nums always equal 1
            item = [1]  # always 1 at start of item
            for i in range(len(final[-1])-1):
                item.append(final[-1][i] + final[-1][i+1])
            item.append(1)  # always 1 at the end of item
            final.append(item)

        return final 



# print(Solution().generate(30))


# # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
#         # need to check every possible root to leaf node combination until finding the sum then could exit traversal (or not finding sum, which means complete traversal)
#         # for every leaf node, check its upstream values vs the targetSum
#         # would maybe need 2 same level fns or nested fn
#         return self.improved(root, targetSum, 0)

#         # SOLUTION: sum each node, pass that sum to children, when reaching base case/leaf node, if sum = targetSum, return True, if any leaf is True, final output is True else False
#     def improved(self, root, targetSum, accSum):
#         if not root: return False 
#         elif not root.left and not root.right:  # means this is a leaf node
#             return accSum + root.val == targetSum
#         # now the main recursive code
#         left = self.improved(root.left, targetSum, accSum + root.val)
#         right = self.improved(root.right, targetSum, accSum + root.val)
#         # print(f'node: {root.val}, {left}, {right}')
#         return left or right 




# root = TreeNode(1)
# root.left = TreeNode(2)
# root.right = TreeNode(3)
# root.right.left = TreeNode(4)
# root.right.right = TreeNode(6)
# print(Solution().hasPathSum(root, 3))




# # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def minDepth(self, root) -> int:
#         # most efficient way would exit as soon as finding the shortest node
#         # need to compare to other nodes to find shortest
#         # recursive fn by its definition not sure if can do the exit
#         # normal recursive fn will run through every node then return values
#         # there is a special case if a node has either left or right, then don't take min
#         if not root:
#             return 0
#         left, right = self.minDepth(root.left), self.minDepth(root.right)
#         # if this is leaf node, return 1 level to upper fn
#         # else means not leaf node, 
#         # should return minimum of left and right, as long as they're not null nodes.
#         # if either is a null node, ignore it and continue with other side to obtain min
#         if not root.left:
#             return 1 + right 
#         elif not root.right:
#             return 1 + left 
#         return 1 + min(left, right) 
        

# root = TreeNode(1)
# root.left = TreeNode(2)
# root.right = TreeNode(3)
# root.right.left = TreeNode(4)
# # print(Solution)
# print(Solution().minDepth(root))



# # Balanced binary tree
# # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:
#     def isBalanced(self, root) -> bool:
#         # A height-balanced binary tree is a binary tree in which the depth of the two subtrees of every node never differs by more than one.
#         def dfs(root):
#             # need to check for both if the subtree is balanced and for its height
#             if not root: return [True, 0]
#             left, right = dfs(root.left), dfs(root.right)
#             # check if this node's left and right subtrees are balanced, return True if so
#             balanced = left[0] and right[0] and abs(left[1] - right[1]) <= 1
#             return [balanced, 1 + max(left[1], right[1])]
#         return dfs(root)[0]



# root = TreeNode(3)
# root.left = TreeNode(9)
# root.right = TreeNode(20)
# root.right.left = TreeNode(15)
# root.right.right = TreeNode(7)
# root.right.right.right = TreeNode(7)
# print(Solution().isBalanced(root))


# def is_age_diverse(lst): 

#     if len(lst) < 10: return False 

#     seen = set()

#     for p in lst:
#         age = p.get('age')
#         # if value between 100 and 200, set as 10 in seen if not in seen
#         if 100 <= age < 200:
#             seen.add(10)
#         # elif value between 10 and 99, set as // 10 in seen if not in seen
#         elif 10 <= age < 100:
#             seen.add(age // 10)
#         # else return false

#     return all(num in seen for num in range(1,10))

# list1 = [
#   { 'firstName': 'Harry', 'lastName': 'K.', 'country': 'Brazil', 'continent': 'Americas', 'age': 29, 'language': 'Python' },
#   { 'firstName': 'Kseniya', 'lastName': 'T.', 'country': 'Belarus', 'continent': 'Europe', 'age': 29, 'language': 'JavaScript' },
#   { 'firstName': 'Jing', 'lastName': 'X.', 'country': 'China', 'continent': 'Asia', 'age': 39, 'language': 'Ruby' },
#   { 'firstName': 'Noa', 'lastName': 'A.', 'country': 'Israel', 'continent': 'Asia', 'age': 40, 'language': 'Ruby' },
#   { 'firstName': 'Andrei', 'lastName': 'E.', 'country': 'Romania', 'continent': 'Europe', 'age': 59, 'language': 'C' },
#   { 'firstName': 'Maria', 'lastName': 'S.', 'country': 'Peru', 'continent': 'Americas', 'age': 60, 'language': 'C' },
#   { 'firstName': 'Lukas', 'lastName': 'X.', 'country': 'Croatia', 'continent': 'Europe', 'age': 75, 'language': 'Python' },
#   { 'firstName': 'Chloe', 'lastName': 'K.', 'country': 'Guernsey', 'continent': 'Europe', 'age': 88, 'language': 'Ruby' },
#   { 'firstName': 'Viktoria', 'lastName': 'W.', 'country': 'Bulgaria', 'continent': 'Europe', 'age': 98, 'language': 'PHP' },
#   { 'firstName': 'Piotr', 'lastName': 'B.', 'country': 'Poland', 'continent': 'Europe', 'age': 128, 'language': 'JavaScript' }
# ]

# print(is_age_diverse(list1))

# # math problem
# def zeros(n: int) -> int:
#     if n == 0: return 0
#     elif n == 1: return 2
#     # for anything greater than len 2, always first number is 1. so all possible combinations of n-1 for 0 and 1. 
#     # 3: 100, 101, 110, 111. =4. 4 - 1. 
#     # 4: 1000, 1001, 1010, 1100, 1011, 1110, 1101, 1111. =8. 8 - 3. 
#     # 5: =16. 10000, 10010, 10011, 10001, 11001, 11100, 11000, 10100
#     # raw answer should be 2 ** n-1. but need to remove any nums with consecutive zeroes. should be a way to calculate how many of those. 
#     # return 2 ** (n - 1)
#     # pattern 2=4-1, 3=8-3, 4=16-7, 5=32-15, 6=64-31, 7=128-63
#     # recursive
#     return 8 * .375
# print(zeros(4))


# # https://www.codewars.com/kata/577e277c9fb2a5511c00001d/train/python
# def vowel_shift(text, n):
#     # need to somehow store the ref to each vowel index. then move each of those according to n
#     # could create a list and a dict. list to trace index position of vowels to use when mapping changes with n, dict to store the vowel at a particular index position.
#     list_idx = []
#     dict_vowels = {}
#     for vowel_i, c in enumerate(text):
#         if c in 'aeiou':
#             list_idx.append(vowel_i)
#             dict_vowels[vowel_i] = c 
#     return 13 % 3

# # [2, 5, 8, 11, 15]
# # {2: 'i', 5: 'i', 8: 'a', 11: 'e'}

# print(vowel_shift('This is a test!', 3))

# def format_words(words):
#     # correct format
#     # if 1 non-null str then just that word
#     # if 2 non-null str, then join w/ 'and'
#     # if 3+ non-null str, then all but last two words joined with ','
#     # may edge cases: empty string in any position
#     # need to clean out null values first to know how many words and then be able to join those words with commas and 'and'
#     final = ''
#     if not words:
#         return final 
    
#     words = tuple(filter(None, words))
#     # if len(words) == 1, then just word, if == 2 then just and, if > 2 then comma all before and...
#     # use ',' and 'and' depending on len
#     for i, word in enumerate(words):
#         # if i is two or less than len(words), join with ','
#         if len(words) - i > 2:
#             final += word + ', '
#         # if i is one less than len(words), join with ' and '
#         elif len(words) - i == 2:
#             final += word + ' and '
#         elif i == len(words)-1:
#             final += word 
#     return final 

# print(format_words(['', 'one', '', 'two', 'three', '']))



# # Input: "gdfgdf234dg54gf*23oP42"
# # Output: "54929268" (because 23454*2342=54929268)

# # WAS MISSING THE '.' WHICH WAS FOR DECIMALS...always read problem (though sometimes it's badly redacted)
# def calculate_string(s):
#     # need to return only digits from left side of operator, by operator, by only digits on right side
#     # operator can be any of +-*/
#     # if there is a "." then ignore everything after for left and right. in case of left, just reset at the operator
#     # do need to iterate through entire str at least once
#     left = ''
#     right = ''
#     operator = ''
#     dot_found = False 
#     for c in s:
#         if not operator:  # add to left str, not right
#             if c in ('+-/*'):  # add operator to variable, reset dot found to false
#                 operator = c 
#                 dot_found = False 
#             elif c == ".":  # if '.' then ignore rest of left side
#                 dot_found = True 
#             elif c.isnumeric() and not dot_found:  # add digits to left side
#                 left += c 

#         elif operator:
#             if c == ".":  # if '.' then ignore rest of left side
#                 break  # finish the loop
#             elif c.isnumeric():  # add digits to left side
#                 right += c 

#     if operator == '+':
#         return str(round(int(left) + int(right)))
#     elif operator == '-':
#         return str(round(int(left) - int(right)))
#     elif operator == '*':
#         return str(round(int(left) * int(right)))
#     elif operator == '/':
#         return str(round(int(left) / int(right)))
                

# print(calculate_string("gdfgdf23.4dg54gf/2.3oP42"))


# def string_constructing(a, s):
#     # need to start with an empty string
#     # the only two operations possible are appending the a string or deleting any char in new string
#     if not s: return 0

#     final = a
#     count = 1
#     pointer = 0

#     while final != s:
#         # can only append the string 'a'
#         # can remove any char from the final string
#         # basically, need to check if final str == s string up to certain index in final, and remove subsequent chars if they do not match and keep checking
#         # you remove when a char is not equal in both strings at same index
#         # you add when all chars in final == s
#         print('here')
#         if len(final) < len(s) and final == s[0:len(final)]:
#             print(final)
#             final += a
#             count += 1
#         elif len(final) > len(s) or final[pointer] != s[pointer]:
#             print(final)
#             final = final[0:pointer] + final[pointer+1:]
#             count += 1
#             pointer += 1


#     return count

# print(string_constructing('aba', 'abbabba'))



# def largest_radial_sum(arr, d):
#     # len(arr) gives the total members
#     # d gives the members by group
#     # n / d gives the total number of groups to compare, which is also step size
#     # could check each iter for loop
#     groups = {}

#     for i in range(len(arr)):
#         key = i % (len(arr) // d)
#         groups[key] = groups.get(key, 0) + arr[i]

#     return max(groups.values())

# print(largest_radial_sum([1,2,3,4], 2))


# def single_digit(n):

#     def bin_sum(num):
#         return sum(int(x) for x in format(num, 'b'))

#     total = bin_sum(n)

#     while total > 9:
#         total = bin_sum(total)

#     return total

# print(single_digit(0))



# # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def sortedArrayToBST(self, nums: list[int]):
#         # should take the midpoint and make it the root
#         # then all values to left will be root.left nodes
#         # all values to right root.right nodes
#         # this should keep at most 1 height difference
#         if not nums:
#             return None

#         m = len(nums) // 2
#         root = TreeNode(nums[m])
#         root.left = self.sortedArrayToBST(nums[0:m])
#         root.right = self.sortedArrayToBST(nums[m+1:])
#         return root


# print(Solution().sortedArrayToBST([-10,-3,0,5,9]))  # always sorted asc


# def solution(n):
#     mappings = {
#         4: 7,
#         7: 4
#     }

#     return mappings.get(n, 0)

# print(solution(4))

# # return the fibonacci seq number at index i
# def fib(i):
#     # fib goes up..no way to no backward nums.
#     if i <= 2:
#         return 1
#     return fib(i-1) + fib(i-2)

# print(fib(6))


# # create a factorial fn that takes in a positive int -> sum product of int - 1 until reaching base case which is 1
# def factorial(num: int) -> int:
#     """takes in a positive integer n and returns the product of all integers between n down to 1."""
#     if num < 1:  # reached base case, return 1 since
#         return 1
#     return num * factorial(num - 1)


# print(factorial(-6))


# # symmetric tree
# # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def isSymmetric(self, root) -> bool:
#         # basically, if you go down each side of the root, if there is a None value on one side and any none-None value on the other, it's False, else True
#         # need to somehow check each subtree's equivalent value on the other side.
#         def dfs(left, right):
#             if not left and not right:  # means both nodes are null, so true
#                 return True
#             elif not left or not right:  # means one node is null, the other is not, false
#                 return False
#             elif left.val == right.val:  # means both nodes have values, if values are equal then True else False
#                 return dfs(left.left, right.right) and dfs(left.right, right.left)
#         return dfs(root.left, root.right)

# root = TreeNode(1)
# root.left = TreeNode(2)
# root.right = TreeNode(2)
# root.left.left = TreeNode(3)
# root.right.right = TreeNode(3)
# root.left.right = TreeNode(4)
# root.right.left = TreeNode(4)
# print(Solution().isSymmetric(root))

# # binary tree sum
# # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def inorderTraversal(self, root) -> list[int]:
#         # in-order seems to be from left-most to right-most
#         # how do i create a list that persists through recursion?
#         if not root:
#             return 0

#         return root.val + self.inorderTraversal(root.left) + self.inorderTraversal(root.right)


# root = TreeNode(2)
# root.left = TreeNode(3)
# root.right = TreeNode(4)
# root.left.left = TreeNode(5)
# root.left.right = TreeNode(6)
# print(Solution().inorderTraversal(root))



# # binary tree inorder traversal
# # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def inorderTraversal(self, root) -> list[int]:
#         # in-order seems to be from left-most to right-most
#         # how do i create a list that persists through recursion?
#         if not root:
#             return []

#         return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
#         # final = 0

#         # def inorder(root):
#         #     if not root:
#         #         return 0

#         #     inorder(root.left)
#         #     final.append(root.val)
#         #     inorder(root.right)

#         # inorder(root)
#         # return final


# root = TreeNode(1)
# root.right = TreeNode(2)
# root.right.left = TreeNode(3)
# print(Solution().inorderTraversal(root))


# def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
#     """
#     Do not return anything, modify nums1 in-place instead.
#     """
#     for i in range(n):
#         nums1[m+i] = nums2[i]

#     nums1.sort()
#     print(nums1)


# print(merge([2,2,3,0,0,0], 3, [1,5,6], 3))


# # Remove duplicates from sorted list
# # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# class Solution:
#     def deleteDuplicates(self, head):
#         # head is the first node
#         # while self, check if self.val == self.next.val and if so, remove self.next from list while connecting self.next.next to self...if self.next is None then finish loop
#         curr = head
#         while curr and curr.next:
#             if curr.val == curr.next.val:
#                 curr.next = curr.next.next
#             else:
#                 curr = curr.next
#         return head


# print(Solution().deleteDuplicates(ListNode(2)))


# def climbStairs(n: int) -> int:
#     # 2 = 2
#     # 3 = 3
#     # 4 = 5 ;;  1,1,1,1 ; 2,2 ; 1,1,2 ; 1,2,1 ; 2,1,1
#     # 5 = 8  ;;  1,1,1,1,1 ; 1,1,1,2 ; 1,1,2,1 ; 1,2,1,1 ; 2,1,1,1 ; 2,2,1 ; 2,1,2 ; 1,2,2
#     # 6 = 13  ;;  1,1,1,1,1,1 ; 1,1,1,1,2 ; 1,1,1,2,1 ; 1,1,2,1,1 ; 1,2,1,1,1 ; 2,1,1,1,1 ; 2,2,2 ; 2,2,1,1 ; 2,1,2,1 ; 1,2,2,1 ; 2,1,1,2 ;
#     return

# print(climbStairs(5))
# print(round(5.432454,2))

# def mySqrt(x):
#     # return sqrt rounded down to the nearest integer. so 2.8 would be 2
#     # 9, 16, 25
#     left=0
#     right=x
#     while left<=right:
#         mid=left+(right-left)//2
#         if x==mid*mid:
#             return mid
#         elif x<mid*mid:
#             right=mid-1
#         else:
#             left=mid+1
#     return left-1

# print(mySqrt(-98))  # 2


# def yields(soap):
#     for num in soap:
#         yield num

# soap = [513]

# for n in yields(soap):
#     print(n)


# def plusOne(digits):
#     # 1st approach is to convert the digits array to an int, add + 1 to that integer, then separate that new integer back into an array...
#     # to implement this, would need to turn into str at some point?
#     num_str = ''.join(str(num) for num in digits)  #
#     new_num_str = str(int(num_str) + 1)
#     return [int(num) for num in new_num_str]

# print(plusOne([9,9,9,9]))


# def lengthOfLastWord(s):
#     return s.split()

# print(lengthOfLastWord('  Hello World how are you   '))


# class Solution:
#     def searchInsert(self, nums: list[int], target: int) -> int:
#         # sorted asc and unique values always
#         # will need to do binary search.
#         if not nums or target < nums[0]: return 0
#         elif target > nums[-1]: return len(nums)
#         l = 0
#         r = len(nums)-1
#         while l <= r:
#             m = l + (r-l)//2  # biased to lower end
#             print('midpoint is now ', nums[m], ' at index ', m )
#             if target == nums[m]:
#                 print('target: ', target, ' equals midpoint: ', nums[m])
#                 return m
#             elif target > nums[m]:
#                 print('target: ', target, ' greater than midpoint: ', nums[m])
#                 l = m + 1
#             else:
#                 print('target: ', target, ' less than midpoint: ', nums[m])
#                 r = m - 1

#         # if l >= r, means midpoint was not found, return l
#         return l

# print(Solution().searchInsert([1,3,5,6], 2))


# def strStr(haystack, needle):
#     for i in range(len(haystack)-len(needle)+1):
#         if haystack[i:i+len(needle)] == needle:
#             return i
#     return -1
    # if len(needle) > len(haystack):
    #     return -1
    # for i in range(len(haystack)):
    #     # if c == c then start checking all needle chars from that point
    #     if haystack[i] == needle[0]:
    #         compare = ''
    #         j = i
    #         n_i = 0
    #         while n_i < len(needle) and j < len(haystack): # stop index errors
    #             if haystack[j] == needle[n_i]:
    #                 compare += haystack[j]
    #                 j += 1
    #                 n_i += 1
    #             else: # there is no match so reset or break while loop
    #                 break
    #         if compare == needle:
    #             return i
    # return -1

# print(strStr('badsad', 'sad'))


# # Longest common prefix
# def longestCommonPrefix(strs):
#     # while index < shortest word length, check each letter in all words if all of them match, then add to prefix, else return prefix
#     # could use the shortest word as base length to iterate through
#     shortest_length = min(len(word) for word in strs)
#     #
#     index = 0
#     prefix = ""
#     # while index is less than shortest word, check if each letter for word[index] for all words is the same and if so, add to prefix
#     while index < shortest_length:
#         for i in range(1, len(strs)):
#             if strs[i][index] != strs[i-1][index]:
#                 return prefix
#         prefix += strs[i][index]
#         index += 1
#     return prefix

# print(longestCommonPrefix(['flower', 'flight', 'floor', 'fly'])) # fl


# # Sample dictionary with soccer player data
# soccer_players = {
#     'Messi': {'goals_scored': 700, 'year_started': 2004, 'year_ended': 2024},
#     'Ronaldo': {'goals_scored': 750, 'year_started': 2002, 'year_ended': 2024},
#     'Neymar': {'goals_scored': 300, 'year_started': 2009, 'year_ended': 2024},
#     'Mbappe': {'goals_scored': 200, 'year_started': 2015, 'year_ended': 2024}
# }

# # Sorting the dictionary by the number of goals scored and year started
# # sorted_players = sorted(soccer_players.items(), key=lambda y: (y[1]['year_started'], y[1]['goals_scored']))
# sorted_players = sorted(soccer_players.keys(), key=lambda y: -soccer_players[y]['goals_scored'])

# # Displaying the sorted dictionary
# for data in sorted_players:
#     print(data)




# class Solution:
#     def romanToInt(self, s: str) -> int:
#         # basically, somehow use ranges or % operand to check edge cases
#         # any letter should represent a number to add, except when an edge case is met, that is the letter's preceding char is I or X or C followed by its respective edge case
#         edge = {
#             'V': 'I',
#             'X': 'I',
#             'L': 'X',
#             'C': 'X',
#             'D': 'C',
#             'M': 'C',
#         }
#         values = {
#             'I': 1,
#             'V': 5,
#             'X': 10,
#             'L': 50,
#             'C': 100,
#             'D': 500,
#             'M': 1000,
#         }
#         total = 0
#         seen = False
#         for i in range(1, len(s)):
#             print(s[i])
#             if seen:
#                 seen = False
#                 continue
#             if s[i-1] in edge.values() and edge.get(s[i], s[i]) == s[i-1]:
#                 # then edge case, so add s[i] and subrtact s[i-1]
#                 total += values[s[i]] - values[s[i-1]]
#                 seen = True
#             else:
#                 total += values[s[i-1]]
#             print(total)
#         return total

# print(Solution().romanToInt('MCMXCIV'))


# def add(*args):
#     product = []
#     for i in range(len(args)):
#         product.append((i+1)*args[i])
#         print(product)
#     return sum(product)

# print(add(1,2,3))


# def corner_fill(square):
#     # should keep track of which way brush stroke is going
#     # first part will always go through all numbers in first list left to right
#     # then down the last index of list through other lists
#     # then once reaches end, move to index - 1 to the left
#     # then go up the lists through same index
#     # then once reach base list, go left through its indexes until list[0]
#     # then go down same index list[0] but for next list to the right
#     # PROBLEM DOESNT SAY IF NUMBERS ARE ALL UNIQUE?
#     # I WILL ASSUME NUMBERS ARE UNIQUE AT FIRST
#     if not square: return []

#     arr = 0
#     i = 0 # start at first index
#     final = []

#     # while the pointer p value is not the last list's first index 0 which is the end value
#     while arr != square[len(square)-1] and i != 0:
#         right = True
#         if right:
#             # if right then go right and down until last list
#             pass
#         else:
#             # if left then go up and left until reaching a list[0]
#             pass

#     return final

# print(corner_fill(
#     [
#         [4,  1, 10,  5],
#         [7,  8,  2, 16],
#         [15, 14, 3,  6],
#         [11, 9, 13, 12]
#     ]
# ))


# def maskify(cc):
#     # if c not in last 4 iterations, change to '#' char, else leave last 4 chars as is
#     mask_length = len(cc) - 4
#     masked = '#' * mask_length if mask_length > 0 else ''
#     return masked + cc[-4:]

# print(maskify('nanananananannana batman!'))


# def switch_lights(arr):
#     # takes too long prev solution find better one
#     # maybe track the sum and reduce its amount each iteration
#     remaining_sum = sum(arr)
#     for i, n in enumerate(arr):
#         if remaining_sum % 2:
#             # if remainder, then this number will switch
#             # remaining sum should be inclusive of this i iteration, change at end
#             # if n == 1:
#             #     arr[i] = 0
#             # else:
#             #     arr[i] = 1
#             arr[i] ^= 1 # this is same as arr[i] = arr[i] ^ 1 which is an XOR bit operation that checks if the bit (arr[i]) is 1, switches to 0 and vice versa for arr[i] == 0 turns to 1
#         remaining_sum -= n
#     return arr

#     # # num at index should change sum(arr[i:]) times, so if that is even, no change, if odd, switch num from 0 to 1 or vice versa
#     # for i, n in enumerate(arr):
#     #     if sum(arr[i:]) % 2:
#     #         # if remainder, means number should change to opposite
#     #         if n == 1:
#     #             arr[i] = 0
#     #         else:
#     #             arr[i] = 1
#     # return arr

# print(switch_lights([0,0,1,0,1]))



# numbers = [1, 2, 3, 4, 5]
# square_generator = (x**2 for x in numbers)

# # Accessing values one at a time using iteration
# for square in square_generator:
#     print(square)
# print(square)


# def candies(lst):
#     return len(lst) * max(lst) - sum(lst) if len(lst) > 1 else -1

# print(candies([5,8,6,4]))


# def find_short(s):
#     # counter = len(s.split()[0] if s else 0)
#     # for word in s.split():
#     #     if len(word) < counter:
#     #         counter = len(word)
#     # return counter

#     return min(len(word) for word in s.split())

# print(find_short("bitcoin take over the world maybe who knows perhaps"))
# # print(find_short(""))


# def to_jaden_case(string):

#     return ' '.join(word.capitalize() for word in string.split())

# print(to_jaden_case("How can mirrors be a real if our eyes aren't real"))


# def square_color(file, rank) -> str:
#     # the specific pattern of 8x8 chessboard is always the same
#     # a1 is black, an so is every square 2 spaces from it
#     # combination of file and rank in terms of if they're even or odd should give answer
#     # looks like all even results for file + rank are black
#     ordinal = 96
#     file = ord(file.lower()) - ordinal
#     return 'Error' if (file not in range(1,9) or rank < 1 or rank > 8) else 'white' if (file + rank) % 2 else 'black'


# print(square_color('g', 0))


# def filter_list(l):
#     return [it for it in l if type(it) is not str ]

# print(filter_list([1,2,'aasf','1','123',123]))


# def maximum_product(arr):
#     # should be a pattern where you account for count of - numbers and + numbers...also acct for 0
#     # if all nums < 0,
#         # if len(list) is even
#             # return the lowest num
#         # else:
#             # return highest num
#     # if all nums > 0,
#         # return lowest num
#     #

# print(maximum_product([-1, -2, -3, -4]))


# def ones_counter(inp):
#     counter = 0
#     final = []
#     for i in range(len(inp)):
#         if inp[i] == 1:
#             counter += 1
#             if i+1 == len(inp):
#                 final.append(counter)
#         elif inp[i] == 0:
#             # check if counter > 0, append to final, reset count to 0
#             if counter > 0:
#                 final.append(counter)
#                 counter = 0
#     return final

# print(ones_counter([1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]))


# import asyncio

# async def asynchronous_generator():
#     for i in range(5):
#         await asyncio.sleep(1)  # Simulate asynchronous operation
#         yield i

# # Usage
# async def main():
#     async for value in asynchronous_generator():
#         print(value)

# asyncio.run(main())



# # Decorator definition
# def my_decorator(func):
#     def wrapper():
#         print("Something is happening before the function is called.")
#         func()
#         print("Something is happening after the function is called.")
#     return wrapper

# # Applying the decorator
# @my_decorator
# def say_hello():
#     print("Hello!")

# # Calling the decorated function
# say_hello()



# class HTMLGen():
#     # how can i automate this as much as possible?
#     # would need to create

#     def __getattribute__(self, name):
#         # return lambda s: f"<{name}>{s}</{name}>"
#         # def wrapper(s):
#         #     return f"<{name}>{s}</{name}>" if name != 'comment' else f"<!--{s}-->"
#         # return wrapper
#         # return lambda name: name
#         return object.__getattribute__(self, name)

# print(HTMLGen().whatever('test here'))


# class Game():

#     def __init__(self, board):
#         self.board = board

#     def play(self):
#         # will need 1 leap for every connected block of 1's.
#         # a connected block of 1's consists of all consecutive 1's which are aligned either horizontally or vertically, not diagonally.
#         leaps = 0
#         seen = set() # add the i and j as one single value here to check
#         for i, arr in enumerate(self.board):
#             # if first or last iteration, only check bottom/top nums, if first or last num, only check right/left
#             for j, num in enumerate(arr):
#                 # will need to check each position for left, right, up, down
#                     # yes, so basically, check this num, if 1 and not in seen set, leaps += 1, add to seen as well as any other adjacent nums
#                 if num == 1:
#                     # if any adj num = 1 and seen, don't add to leaps, but add all adj nums that = 1 to seen
#                     if (i,j) not in seen: # so num = 1 and prev seen
#                         seen.add((i,j))
#                         leaps += 1
#                     try:
#                         if self.board[i-1][j] == 1 and self.board[i-1][j] not in seen:
#                             seen.add((i-1,j)) # up
#                     except: pass
#                     try:
#                         if self.board[i+1][j] == 1 and self.board[i+1][j] not in seen:
#                             seen.add((i+1,j)) # down
#                     except: pass
#                     try:
#                         if self.board[i][j-1] == 1 and self.board[i][j-1] not in seen:
#                             seen.add((i,j-1)) # left
#                     except: pass
#                     try:
#                         if self.board[i][j+1] == 1 and self.board[i][j+1] not in seen:
#                             seen.add((i,j+1)) # right
#                     except: pass
#                 # print(seen)
#         print(leaps)
#         return leaps


# g = Game(
#     [
#         [1,0,0,0,0],
#         [0,0,1,1,0],
#         [1,0,1,0,1],
#         [1,1,1,1,0],
#         [1,1,1,0,1],
#     ]
# ) # 4

# g.play()


# class State:
#     # Implement the State class here
#     def __init__(self):
#         # should initialize as unauthorized, then change if login/logout
#         self.state = 'unauthorized'

#     def __repr__(self) -> str:
#         return self.state


# def create_paginator(items: list, pageSize: int) -> list:
#     for i in range(0, len(items), pageSize):
#         yield items[i:i+pageSize]


# print(create_paginator([1,2,3,4,5,6], 2))


# def solution(text, ending):
#     # your code here...
#     if len(ending) > len(text): return 'Ending cannot be greater than text.'
#     end_length = len(ending)
#     return ending in text[-end_length:]

# print(solution('samurai', 'murai'))


# def xo(s):

#     xes = 0
#     oes = 0
#     for c in s.lower():
#         if c == 'x': xes += 1
#         elif c == 'o': oes += 1
#     return xes == oes
#     # return s.count('x') == s.count('o')

# print(xo("xoOxo"))


# def solve(s):
#     #
#     # if len(s) <= 1:
#     #     return 0
#     # prefixes = []
#     # suffixes = []
#     # for i in range(1,len(s)):
#     #     prefixes.append(s[:i])
#     # for i in range(1,len(s)):
#     #     suffixes.append(s[i:])
#     # greatest = 0
#     # for pre in prefixes:
#     #     if pre in suffixes and len(pre)*2 <= len(s):
#     #         if len(pre) > greatest:
#     #             greatest = len(pre)
#     # return greatest if greatest else 0

#     largest = 0
#     for i in range(1, len(s)//2+1):
#         if s[:i] == s[-i:]:
#             largest = i
#     return largest if largest else 0


# print(solve('aaaaaa'))


# def hex_word_sum(s):
#     # if word contains letter that is not A-F or O or S, return 0
#     # O == 0, S == 5

#     return hex(ord('F'))

# print(hex_word_sum('SAFE'))


# def get_percentage(sent, limit=1000) -> str:
#     # return a rounded integer number that represents sent/limit as a percentage ie 10% as a string
#     # if sent == 0, 'No e-mails sent'
#     # if sent >= limit, 'Daily limit is reached'
#     return 20 * 5 / 4
#     if sent == 0:
#         return 'No e-mails sent'
#     elif sent >= limit:
#         return 'Daily limit is reached'
#     else:
#         percentage = int(sent / limit * 100)
#         return f"{percentage}%"

# print(get_percentage(66.6, 100))


# def mirror(data: list[int]) -> list:
#     # need to get unordered list as input, then order the list both ways and extend/add both lists together except for the first value of desc list (which is the greatest value and should only be in list once)
#     # asc = sorted(data)
#     # desc = sorted(data, reverse=True)[1:]
#     # return asc + desc
#     return sorted(data) + sorted(data, reverse=True)[1:]

# print(mirror([-5,10,8,10,2,-3,10]))


# def version_compare( version1, version2 ):
#     # from left to right, if number in same decimal position is higher, then that version is greater
#     # input will be 2 strings to compare of maybe equal or different lengths and decimal places
#     # can convert all versions to 5 part versions (just add 0 if none exist) and compare equally
#     version1_test = version1.split('.')
#     version2_test = version2.split('.')
#     if len(version1_test) < 5:
#         # add '.0' for however decimals are missing
#         version1 += '.0' * (5 - len(version1_test))
#     if len(version2_test) < 5:
#         # add '.0' for however decimals are missing
#         version2 += '.0' * (5 - len(version2_test))
#     # have two equal length in terms of decimals strings
#         print(version1)
#         print(version2)
#     version1 = version1.split('.')
#     version2 = version2.split('.')
#     for i, num in enumerate(version1):
#         if int(num) > int(version2[i]):
#             return 1
#         elif int(num) < int(version2[i]):
#             return -1
#     return 0

# print(version_compare('2', '2.0'))


# def calculate_time(p1, p2):
#     # if p1[0] != p2[0] then calculate difference factor within those 5 seconds bw p1, p2 to get total
#     # else, do the same but for p1[1], p2[1]
#     interval = 5
#     if p1[0] != p2[0]:
#         return round((abs(p2[0]) / (abs(p1[0]) - abs(p2[0])) * interval), 3)
#     else:
#         return round((abs(p2[1]) / (abs(p1[1]) - abs(p2[1])) * interval), 3)

# print(calculate_time([50, -100], [47.5, -95]))


# def find_the_missing_tree(trees):
#     # could create dict with values as keys and count of that value as values
#     # could also use the count() fn using a set, but would need to traverse list multiple times, one for each number, not the most time efficient
#     counts = {}

#     # return sorted(trees, key=trees.count)
#     for num in trees:
#         if num in counts.keys():
#             counts[num] += 1
#         else:
#             counts[num] = 1

#     for k,v in counts.items():
#         if v == min(counts.values()):
#             return k


# print(find_the_missing_tree([11, 2, 3, 3, 3, 11, 2, 2]))


# '''
# need to create the following:
# - an online library where a user can add or remove books from their library.
# - each book has a title and its content.
# - need to keep track of last page read in any of the user's books in library
# - need to keep track of which is the currently active book, all others are inactive


# ignore User class, just think about how a single user would see this online library
# Book class:
#     - id
#     - title
#     - text content
#     - bookmark
#     - active

# Library class:
#     - books

# '''

# class Book:
#     def __init__(self, id: int, title: str, text: str, active: bool=False):
#         self.id = id
#         self.title = title
#         self.text = text
#         self.active = active

#     def get_bookmark(self, last_page: int):
#         self.last_page = last_page
#         return last_page

#     def __repr__(self) -> str:
#         return f"<{self.title} | {self.text[:15]}...>"

# class Library:

#     def __init__(self, id, name):
#         self.id = id
#         self.name = name
#         self.books = {}

#     def add_to_library(self, book):
#         # add the book id as a key, and book object as value into books
#         self.books[book.id] = book
#         return self.books

#     def remove_from_library(self, id):
#         print(f"REMOVING: {self.books.pop(1)}")
#         return self.books


# book = Book(1, 'The Hobbit', 'Chapter 1: In a hole in the ground...')
# l = Library(1, 'fiction')
# print(l.add_to_library(book))
# l2 = Library(2, 'nonfiction')
# print(l2.books)


# def convert_to_mixed_numeral(parm):
#     # both input and output are strings
#     # will be given improper fraction which can be
#     numbers = parm.split('/')
#     num, den = abs(int(numbers[0])), int(numbers[1]) # doubt here unsure if negative sign is taken into acct in int conversion...
#     # maybe not require conversion (if num < den)
#     if num < den: return parm
#     # if can be converted to whole number, then whole number
#     elif num % den == 0: return (parm[0] if parm[0] =='-' else '') + str(int(num / den))
#     # negative
#     return (parm[0] if parm[0] == '-' else '') + str(num // den) + ' ' + str(num % den) + '/' + str(den)

# print(convert_to_mixed_numeral('-43/21'))


# from itertools import permutations

# # Example 1: Permutations of a list
# my_list = [1, 2, 3, 4]
# perms_list = permutations(my_list)

# # Displaying permutations
# count = 0
# for perm in perms_list:
#     print(perm)
#     count += 1
# print((count))


# def rearranger(k, *args):
#     # need to return lowest possible number by rearranging args that is divisible by k
#     # if multiple rearrangements of args give the same result, include those multiple options, sort them by lowest ie 2, 32 vs 23, 2
#     # how to rearrange? in theory, last num needs to be div by k. but last num could be part of larger num if concat w other args...
#       # solution will be the one that somehow takes all possible short combinations that are div by k, then rest of nums left orders from smallest to largest...
#     # first take the numbers that are div by k (could be concatenated nums)
#     return f"Rearrangement: _ generates: _ divisible by {k}" # single answer
#     return "Rearrangements: " + "_" + " and " + f"generates: _ divisible by {k}" # multiple answers
#     return "There is no possible rearrangement"

# print(rearranger(4, 32, 3, 34, 7, 12)) # 12, 3, 34, 7, 32


# def factorial(n):
#     # will need to multiply number and somehow go backwards from n all the way to 1
#     if n > 1:
#         return n * factorial(n-1) # if n > 1, else ?
#     return 1 # since this is multiplication need a 1

# print(factorial(9)/6)


# def happy_numbers(n):
#     # will need to go n-1 for each round..and for that round's n, calculate if it's a happy number, if so, all numbers in sequence from n to 1 are happy numbers which i could potentially store
#     final_happy_nums = []
#     for i in range(n, 1, -1): # for number from 1 to n in reverse
#         # for number, split its digits, find the sum of the squares of its digits
#         number = i
#         iterations = set()
#         while number != 1 or number not in iterations:
#             digits = list(str(number)) # ['1','0']
#             square_sum = sum(int(d)**2 for d in digits) # 1
#             iterations.add(square_sum) # 1
#             number = square_sum
#             print(number)
#         if number == 1:
#             final_happy_nums.append(number)

#                 # if the sum of squares does not equal 1, set sum as new number and repeat until either sum of squares does equal 1, or one of the past iterations of big scope number within this sequence is equal to this iteration's n. In other words, if 89 has already appeared in this sequence before, and sum of squares in this iter is 89, this is NOT a happy number
#     return final_happy_nums # the happy numbers


# print(happy_numbers(10))


# def alphabet_position(text):
#     # should take a string and replace each letter with its position int he alphabet
#     # rules: space between each number; only upper and lowercase alpha chars allowed
#     # final = []
#     # for c in text.lower(): # - 96
#     #     if c.isalpha():
#     #         final.append(str(ord(c) - 96))
#     # return ' '.join(final)

#     print({{c: str(ord(c) - 96)} for c in text.lower() if c.isalpha()})

#     return (str(ord(c) - 96) for c in text.lower() if c.isalpha())
#     sentence = "This is a sample  n sentence."
#     result = sentence.split()
#     print(result)


# print(alphabet_position("The sunset sets at twelve o' clock."))


# def infected_zeroes(lst):
#     # if 0 is n number in a list, then both n-1 and n+1 in terms of index become infected if they were not before
#     # need to account for edge cases, so lst[0] and lst[-1]
#     # 2 errors: 1. turns dont update correctly since not taking into acct when a number is converted; 2. time out error
#     turns = 0
#     while sum(lst) != 0:
#         left_was_1 = False
#         right_was_1 = False
#         for i,num in enumerate(lst):
#             # if first item, just check its right side
#             if i == 0:
#                 if num == 0 and lst[i+1] == 1:
#                     lst[i+1] = 0
#                     right_was_1 = True
#             # if last item, just check its left side
#             elif i == len(lst)-1:
#                 if num == 0:
#                     lst[i-1] = 0
#             # else check both left and right side
#             else:
#                 # if this num = 0 but it was 1 in previous iter
#                 if num == 0 and right_was_1:
#                     right_was_1 = False
#                 # if num = 0 and not previously 1
#                 elif num == 0 and not right_was_1:
#                     lst[i-1] = 0
#                     if lst[i+1] == 1:
#                         lst[i+1] = 0
#                         right_was_1 = True

#         # end of turn, add 1 to turn
#         turns += 1

#     return turns

# print(infected_zeroes([0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1]))


# def ride(group, comet):
#     # inputs strings, all capital letters, no spaces
#     # convert each letter from each input to number, multiply each string's resulting numbers
#     # divide the final number % 47 and if the resulting remainder is = to other str, return 'GO' else 'STAY'
#     alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     group_prod = 1
#     for c in group:
#         group_prod *= (alpha.index(c) + 1)

#     comet_prod = 1
#     for c in comet:
#         comet_prod *= (alpha.index(c) + 1)

#     if group_prod % 47 == comet_prod % 47:
#         return 'GO'
#     else:
#         return 'STAY'

# print(ride('COMETQ', 'HVNGAT'))


# def get_in_line(queue: list[int]) -> int:
#     # if known guest, all places in line swap, except for other known guests and decoys
#     # order of priority = known guest = 1, member = 2, decoys = 3 AND unkown guests = > 3
#     # you are 0 and an unkown guest
#     # for sorting purposes, decoys and unkown guests numbers should stay in same order once known guests and members are pushed in front of the line

#     # sort the queue by looping through list and if num > 2 then add to new list, if less than two add to another new list, then sort the 1-2 list and add the other new list to the end of that list to get final sorted
#     priority_list = []
#     non_priority_list = []
#     for num in queue:
#         if num == 1 or num == 2:
#             priority_list.append(num)
#         else:
#             non_priority_list.append(num)
#     final = sorted(priority_list) + non_priority_list
#     print(final)

#     # start consuming queue, though if i loop and pop an item, loop will need to restart at 0
#     # if the popped num == 1, swap will need to occur with rest of list, add popped item to list
#     consumed = []
#     for num in range(len(final)):
#         to_add = final.pop(0)
#         consumed.append(to_add)
#         if to_add == 0:
#             return len(consumed)
#         elif to_add == 1: # do swaps following rules
#             for i in range(len(final)//2): #THIS COULD NOT WORK SINCE DONT KNOW IF FINAL HERE GRABS UPDATED LIST OR JUST THE ORIGINAL FINAL LIST...
#                 # if 1 or 3 not on either end, swap
#                 if final[i] not in [1,3] or final[len(final)-1-i] not in [1,3]:
#                     final[i], final[len(final)-1-i] = final[len(final)-1-i], final[i]


# print(get_in_line([0, 8, 2, 1, 4, 2, 12, 3, 2]))

# def closest_power(n):
#     # thinking maybe can take square root of num and grab nearest whole number to that square root (5.8 = 6)
#     # then somehow would need to check instead of sq roots, higher power roots...
#     return n**(1/2)

# print(closest_power(31))


# some = [10,21,4,2143]
# for i in range(len(some)):
#     print(some[i])


# def pattern(number: int) -> str:
#     # will essentially need 2*n-1 lines for this, so for 15 will need 29
#     # each line's length is the same as rows 2*n-1
#     # should always start w/whole numbers in chronological order up, base 9 (10==0)

#     line = list(' '*(number*2 - 1))
#     final = []
#     left = 0
#     right = -1
#     going_down = number
#     for i in range(len(line)):
#         # should have a pointer on both ends changing right/left each iteration
#         # change the value at [0] and [-1]
#         # missing: when reach number (middle) start going down (i-1); 10 should equal 0
#         line[left] = str(i+1)[-1] if i+1 <= number else str(going_down-1)[-1]
#         line[right] = str(i+1)[-1] if i+1 <= number else str(going_down-1)[-1]
#         final.append(''.join(line))
#         left += 1
#         right -= 1
#         if i+1 > number:
#             going_down -= 1
#         line = list(' '*(number*2 - 1))

#     return '\n'.join(final)

# print(pattern(0))


# import re

# def is_audio(file_name):
#     # at least 1 or more alpha chars, no spaces, before a '.' char
#     # a '.' char
#     # a file extension ending after '.' that matches 1 of 4 options
#     match = re.search(r'[A-Za-z]+\.(mp3|flac|alac|aac)', file_name)
#     if match:
#         return len(match[0]) == len(file_name)
#     return False

# def is_img(file_name):
#     match = re.search(r'[A-Za-z]+\.(jpg|jpeg|png|bmp|gif)', file_name)
#     if match:
#         return len(match[0]) == len(file_name)
#     return False


# print(is_audio('Nothing else matters.mp3'))
# print(is_audio('DaftPunk.FLAC'))


# def look_and_say(data, maxlen):

#     result_list = []

#     for n in range(maxlen):
#         look = result_list[-1] if result_list else data
#         # grab latest str from result_list and split each int
#         # 111111856222
#         # need to loop through current look, set a variable to check if consecutive of that variable
#         same_count = 0
#         value = look[0]
#         add_to_result_list = ''
#         for num in look:
#             if value == '':
#                 value = num
#                 same_count += 1
#                 if len(look) == 1:
#                     add_to_result_list += f'{str(same_count)}{value}'
#             elif num == value:
#                 # then keep adding count of that number
#                 same_count += 1
#             # only update if value is not None
#             else:
#                 # then + 1 to same_count, update the add_to_result_list, change the value and reset same_count to 1
#                 print(same_count, value)
#                 same_count += 1
#                 add_to_result_list += f'{str(same_count)}{value}'
#                 # resest
#                 value = num
#                 same_count = 0
#         if not add_to_result_list:
#             add_to_result_list += f'{str(same_count)}{value}'
#         result_list.append(add_to_result_list)
#         # print(result_list)

#     return result_list

# print(look_and_say('1', 7))


# def a1_thick_and_hearty(a1, a2):
#     # first find the set of common elements bw the two arrays
#     # based on that set, check which two numbers within the set either add or subtract to get a total equal to the length of each list. in the end, should have a 2 - 4 number set
#     # i will suppose that only two numbers add or subtract to create the final set
#     matches_set = set()
#     for num in a1:
#         if num in a2:
#             # add matching numbers to set
#             matches_set.add(num)
#     # 1,2,4,6
#     # matches = list(matches_set)
#     final_set = set()
#     for i,n in enumerate(matches_set):
#         for j,n2 in enumerate(matches_set):
#             # if not same number, add and subtract other numbers
#             if i != j:
#                 if n+n2 == len(a1) or n-n2 == len(a1) or n+n2 == len(a2) or n-n2 == len(a2):
#                     final_set.add(n)
#                     final_set.add(n2)
#                     if len(final_set) > 2:
#                         break
#     return final_set
#     # x = set(filter(lambda n: n in a2, a1))
#     # return x

# print(a1_thick_and_hearty([1, 2, 3, 4, 5, 6], [1, 2, 4, 6, 7, 8, 9, 10]))


# def sursurungal(string):
#     # will need to check could be multiple nums + words in the str input
#     to_list = string.split(' ')
#     for i in range(1, len(to_list)):
#         # split at spaces and check if number first and word 2nd that works
#         if to_list[i].isalpha() and to_list[i-1].isnumeric():
#             # singular no marker: 0 or 1 and leave as is
#             # prefix bu: 2  and remove 's' end of word
#             if int(to_list[i-1]) == 2:
#                 to_list[i] = 'bu' + to_list[i][:-1]
#             # suffix zo: 3 to 9 and remove 's' end of word
#             if 3 <= int(to_list[i-1]) <= 9:
#                 to_list[i] = to_list[i][:-1] + 'zo'
#             # both pre/suffix: 10+ and remove 's' end of word
#             if int(to_list[i-1]) > 9:
#                 to_list[i] = 'ga' + to_list[i][:-1] + 'ga'
#     return ' '.join(to_list)


# print(sursurungal('5 lions and 15 zebras and 100 elephants'))


# def digital_root(n):
#     # need to grab the 'n' which is int and take each of its digits, so turn 'n' to str and split at each number into a list and grab its sum in order to check if sum is one digit. if more than one digit repeat until only one digit left.
#     # specify condition when n < 10 to return that number
#     if n < 10:
#         return n
#     else:
#         total_sum = 0
#         for num in (str(n)):
#             total_sum += int(num)
#         return digital_root(total_sum) # need return here to be able to use the return data
#         # return digital_root(sum([int(num) for num in str(n)]))

# print(digital_root(942))


# def is_possible(arr, target_sum):
#     def backtrack(index, current_sum):
#         if index == len(arr):
#             return current_sum == target_sum

#         # Try adding the current element
#         if backtrack(index + 1, current_sum + arr[index]):
#             return True

#         # Try subtracting the current element
#         if backtrack(index + 1, current_sum - arr[index]):
#             return True

#         return False

#     return backtrack(1, arr[0])  # Start with the first element

# # Example usage
# arr = [1, 3, 4, 6, 8]
# target_sum = -2
# result = is_possible(arr, target_sum)
# print(result)


# def unpack_sausages(truck):
#     # input is a list of tuples with different amount of strings per tuple
#     # a valid sausage package contains 4 total sausages, and all 4 are the same, and all four are either
#     pass

# print(unpack_sausages(
#     [
#     ("(-)", "[IIII]", "[))))]"), ("IuI", "[llll]"), ("[@@@@]", "UwU", "[IlII]"), ("IuI", "[))))]", "x"), ()
#     ]
#     ))


# def locate(seq, value):
#     # seq is list or multiple nested lists
#     # value is a str
#     # need to flatten lists all into one
#     def flatten(seq):
#         flat_list = []
#         for item in seq:
#             if isinstance(item, list): # CHECK FOR TUPLE AS WELL
#                 flat_list.extend(flatten(item))
#             else:
#                 flat_list.append(item)
#         return flat_list

#     return value in flatten(seq)

# print(locate(['a','b',['c','d',['e']]],'c'))



# def ranks(a):
#     # need to return ranks highest being 1, ties being ties
#     # 1st option to sort numbers, then assign 1,2,3,etc or same number if repeat
#     a_sorted = sorted(a, reverse=True)
#     final = {}
#     # if repeat, can take next rank of non-repeating num by keeping tab of index
#     for i, n in enumerate(a_sorted):
#         # first num is 1
#         if i == 0:
#             final[str(n)] = 1
#         # after first num, check if num equal to last iteration
#         elif n != a_sorted[i-1]:
#             final[str(n)] = i + 1
#     # return [final[str(n)] for n in a]
#     return [x for x in enumerate(final.items())]

# print(ranks([3,3,3,3,3,5,1])) # [2,2,2,2,2,1,7]


# def locate_entrance(office: list[str]) -> tuple[int, int]:
#     # only options for office entrance:
#     #   first str in array or last str = any '.' char in any of those rows
#     #   any other str in array =
#     #       first or last char is a '.' ignoring any spaces before or after within str
#     #       if char is '.' and either prev or next row at same index is not '#' or if that prev/next row at same index does not exist
#     for floor, row in enumerate(office):
#         #   first str in array or last str = any '.' char in any of those rows
#         if floor == 0 or floor + 1 == len(office):
#             if '.' in row:
#                 return row.index('.'), floor
#         #   any other str in array =
#         #       first or last char is a '.' ignoring any spaces before or after within str
#         elif row.strip()[0] == '.':
#             return row.index('.'), floor
#         elif row.strip()[-1] == '.':
#             return len(row) - 1 - row[::-1].index('.'), floor
#         #       if char is '.' and either prev or next row at same index is not '#' or if that prev/next row at same index does not exist
#         for i, char in enumerate(row):
#             if char == '.':
#                 # if prev/next row is shorter than char index ONLY WORKS FOR RIGHT SIDE, NOT LEFT
#                 if len(office[floor-1]) >= i:
#                     if office[floor-1][i] == ' ':
#                         return i, floor
#                 elif len(office[floor+1]) >= i:
#                     if office[floor+1][i] == ' ':
#                         return i, floor

# print(locate_entrance(
#                         [' #####',
#                         ' #...#',
#                         ' #...#',
#                         ' #...#',
#                         '##.#  ',
#                         '#..###',
#                         '######']
#                     )
# )

# my_list = ['a', 'b', '.', 'c', '.', 'd', 'e', '.']

# # Reverse the list and find the index of the last '.' character
# reversed_index = my_list[::-1].index('.')
# # Convert the reversed index to the original index
# last_dot_index = len(my_list) - 1 - reversed_index

# print("Index of the last '.' character:", reversed_index)



# def squares_to_odd(a: int, b: int) -> str:
#     diff = a**2 - b**2
#     # start_check = diff//2 if diff//2 % 2 == 1 else diff//2 - 1
#     start_check = diff if diff % 2 == 1 else diff - 1
#     for num in range(start_check, -2, -2):
#         total_sum = []
#         print(num)
#         for num2 in range(num, -2, -2):
#             if sum(total_sum) == diff and a-b == len(total_sum):
#                 list_str = ''
#                 for num in sorted(total_sum):
#                     list_str += f'{num} + '
#                 list_str = list_str[:-2] + f'= {diff}'
#                 return f'''{a}^2 - {b}^2 = {list_str}'''
#             elif sum(total_sum) > diff:
#                 break
#             else:
#                 total_sum += [num2]


# print(squares_to_odd(3, 0))


# def alphabetic(s):
#     #
#     return s == ''.join(sorted(s))

# print(alphabetic('abcd'))


# # Example 1
# num = 42
# bit_length = num.bit_length()
# print(f"The bit length of {num} is: {bit_length}")

# # Example 2
# negative_num = -128
# bit_length_negative = negative_num.bit_length()
# print(f"The bit length of {negative_num} is: {bit_length_negative}")



# def bin_rota(arr):
#     # the order of the lists is, if list index is even, leave as is, if odd, reverse list
#     # can either create new list and append to it, or keep adding to an existing list
#     final = []
#     for i, row in enumerate(arr):
#         if i % 2:
#             row.reverse()
#         final.extend(row)
#     return final


# print(bin_rota(
#     [
#         ["Bob","Nora"],
#         ["Ruby","Carl"]
#     ]
# ))


# def queue(queuers: list, pos: int) -> int:
#     # account for position, length of list, nums less than num at position
#     return queuers[pos] * len(queuers) - sum(queuers[pos] - num for num in queuers if queuers[pos] > num) - len(queuers) - pos + 1

# print(queue([2, 5, 3, 6, 4], 1)) # 12


# def boredom(staff: dict) -> str:

#     dept_score = {
#         "accounts": 1,
#         "finance": 2,
#         "canteen": 10,
#         "regulation": 3,
#         "trading": 6,
#         "change": 6,
#         "IS": 8,
#         "retail": 5,
#         "cleaning": 4,
#         "pissing about": 25
#     }

#     total_score = 0

#     for k,v in staff.items():
#         total_score += dept_score[v]
#         # print(v, dept_score[v])

#     if total_score <= 80:
#         return "kill me now"
#     elif total_score < 100:
#         return "i can handle this"
#     else:
#         return "party time!!"

# print(boredom({ "tim": "accounts", "jim": "accounts",
#       "randy": "pissing about", "sandy": "finance", "andy": "change",
#       "katie": "IS", "laura": "IS", "saajid": "canteen", "alex": "pissing about",
#       "john": "retail", "mr": "pissing about" }))


# def two_count(n):
#     if n < 2 or n % 2 != 0:
#         return 0

#     count = 0
#     while n % 2 == 0:
#         n /= 2
#         count += 1
#     return count

# print(two_count(256)) # 4


# def count_salutes(hallway):
#     # if right, will meet with subsequent lefts ONLY
#     # if left, will meet with previous rights ONLY
#     # salute only occurs when right meets left. right always first, to salute it needs a left
#     right = 0
#     salutes = 0
#     for c in hallway:
#         if c == '>':
#             right += 1
#         elif c == '<':
#             salutes += right * 2
#     return salutes

# print(count_salutes('<---<--->----<'))


# def bouncing_ball(h, bounce, window):
#     if h > 0 and bounce > 0 and bounce < 1 and window < h:
#         # ball only seen if > window, else not seen
#         times_seen = 1
#         while h > window:
#             h = h * bounce
#             if h > window:
#                 times_seen += 2
#         return times_seen
#     else:
#         return -1

# print(bouncing_ball(3, 0.66, 1.5)) # 3


# def good_vs_evil(good, evil):
#     # good_points = [1,2,3,3,4,10]
#     evil_points = [1,2,2,2,3,5,10]
#     # # need to multiply points by string according to index
#     # good_total = 0
#     # for i, v in enumerate(good.split(' ')):
#     #     good_total += int(v) * good_points[i]
#     # evil_total = 0
#     # for i, v in enumerate(evil.split(' ')):
#     #     evil_total += int(v) * evil_points[i]

#     # if good_total > evil_total:
#     #     return "Battle Result: Good triumphs over Evil"
#     # elif evil_total > good_total:
#     #     return "Battle Result: Evil eradicates all trace of Good"
#     # else:
#     #     return "Battle Result: No victor on this battle field"
#     evil_int = [int(n) for n in evil.split(' ')]
#     return type(map(lambda p, e: p * e    , evil_points, evil_int))

# print(good_vs_evil('0 0 0 0 0 10', '0 3 1 1 9 0 0'))


# def parse(data):
#     # TODO: your code here
#     updated = 0
#     final = []
#     for c in data:
#         if c == 'i':
#             updated += 1
#         elif c == 'd':
#             updated -= 1
#         elif c == 's':
#             updated **= 2
#         elif c == 'o':
#             final.append(updated)
#     return final

# print(parse("iiisdoso"))


# def up_array(arr):
#     # only positive int, only one digit int
#     # only issue is when num is a 9
#         # need to convert that value to 0, and add 1 to left num
#     # could join all ints into str, then add 1 to that num, then split at every digit...
#     joined_num = ''
#     for d in arr:
#         joined_num += str(d)
#     joined_num = str(int(joined_num) + 1)
#     # final = []
#     # for d in joined_num:
#     #     final.append(int(d))
#     return [int(d) for d in joined_num]

# print(up_array([9,9,9,9]))


# def multiplication_table(size):
#     # need to return a list of lists which goes up to the number 'size'
#         # total num of lists = size, total nums in each list = size
#     # for loop starting at 1 up to size
#     table = []
#     for i in range(1, size+1):
#         row = []
#         for num in range(1, size+1):
#             row.append(i * num)
#         table.append(row)
#     return table


# print(multiplication_table(3))


# def solution(n):
#     # between 1 and 3999
#     # Modern Roman numerals are written by expressing each digit separately starting with the left most digit and skipping any digit with a value of zero.
#     # In Roman numerals 1990 is rendered: 1000=M, 900=CM, 90=XC; resulting in MCMXC. 2008 is written as 2000=MM, 8=VIII; or MMVIII. 1666 uses each Roman symbol in descending order: MDCLXVI.

#     # always checking the highest divisible value first. then go down a level (3999 checks if / 1000)
#     romans = ['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
#     values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
#     final = ''
#     for i, v in enumerate(values):
#         calc = n // v
#         if calc > 0:
#             final += calc*romans[i]
#             n -= calc*v
#     return final

# print(solution(1990))


# def find_nb(m):
#     # recursive fn
#     total = 0
#     n = 1
#     while total < m:
#         total += n**3
#         n += 1
#     return n-1 if total == m else -1

# print(find_nb(135440716410000)) # 3


# def count_smileys(arr):
#     eyes = ':;'
#     nose = '-~'
#     mouth = ')D'
#     counter = 0
#     for smiley in arr:
#         if len(smiley)==3 and smiley[0] in eyes and smiley[1] in nose and smiley[2] in mouth:
#             counter += 1
#         elif len(smiley)==2 and smiley[0] in eyes and smiley[1] in mouth:
#             counter += 1
#     return counter

# print(count_smileys([':)',':(',':D',':O',':;']))


# def comp(array1, array2):
#     if not array1 and array2:
#         return False
#     array1.sort(), array2.sort()
#     for i, n in enumerate(array1):
#         if n**2 != array2[i]:
#             return False
#     return True

# print(comp([121, 144, 19, 161, 19, 144, 19, 11], [11*11, 121*121, 144*144, 19*19, 161*161, 19*19, 144*144, 19*19]))


# def is_prime(num):
#     if num < 2:
#         return False
#     for n in range(2, num//2+1):
#         if num % n == 0:
#             return False
#     return True

# print(is_prime(31))


# def tribonacci(signature, n):
#     #  you need to create a fibonacci function that given a signature array/list, returns the first n elements - signature included of the so seeded sequence.
#     # Signature will always contain 3 numbers; n will always be a non-negative number; if n == 0, then return an empty array
#     # will always be taking prev 3 numbers from list..find way to add
#     for _ in range(n-3):
#         signature.append(sum(signature[-3:]))
#     return signature[:n]

# print(tribonacci([0, 0, 1], 10))


# def abbreviate(s):
#     words = []
#     word = ""

#     for char in s:
#         if char.isalpha():
#             word += char
#         else:
#             if word:
#                 words.append(word)
#                 word = ""
#             words.append(char)

#     if word:
#         words.append(word)

#     for i, word in enumerate(words):
#         if len(word) > 3:
#             words[i] = f'{word[0]}{len(word)-2}{word[-1]}'

#     return ''.join(words)

# print(abbreviate("elephant-ride goes !! round"))


# def deep_count(a):
#     count = 0
#     for obj in a:
#         count += 1
#         if type(obj) is list:
#             count += deep_count(obj)
#     return count

# print(deep_count([1, 2, [3, 4, [5]]])) # 7


# def delete_nth(order,max_e):
#     # need to iterate through list
#     # need to count number of each number
#     final = []
#     for num in order:
#         if final.count(num) < max_e:
#             final.append(num)
#     return final
#     # return [num for num in order if ]

# print(delete_nth([20,37,20,21], 1))


# def high(x):
#     # Code here
#     final = ''
#     highest = 0
#     for word in x.split(' '):
#         score = 0
#         for c in word:
#             score += ord(c)-ord('a')+1
#         if score > highest:
#             highest = score
#             final = word
#     return final

# print(high('man i need a taxi up to ubud'))


# def tower_builder(n_floors):
#     # as you go down pyramid
#         # numbers increase and always start with -> 1,3,5,7,etc
#         # need to insert n_floors -1 whitespaces with each round of additional levels
#     # empty list
#     # start value to include whitespaces
#     # for loop that adds '*' char x n_floors, and reduces start value to reduce whitespaces each turn
#     pyramid = []
#     remaining = n_floors - 1
#     for i in range(1, n_floors+1):
#         prev = i - 1
#         width = i + prev
#         pyramid.append(" " * remaining + "*" * width + " " * remaining)
#         remaining -= 1
#     return pyramid

# print(tower_builder(3))


# def beggars(values, n):
#     # need for loop
#     # need to do steps of n
#     beggars_take = []
#     for beggar in range(n):
#         # print(beggars_take)
#         total = 0
#         if beggar >= n:
#             beggars_take.append(0)
#         else:
#             for take in range(values[beggar], len(values)+1, n):
#                 total += take
#                 # print(take)
#         beggars_take.append(total)
#     return beggars_take

# print(beggars([1,2,3,4,5],7))


# def find_missing_letter(chars):
#     for i in range(len(chars)-1):
#         if ord(chars[i+1]) - ord(chars[i]) != 1:
#             return chr(ord(chars[i])+1)


# print(find_missing_letter(['a','b','c','d','f']))


# def triple_double(num1, num2):
#     #code me ^^
#     # store each digit num1
#     # for next digit num1, if digit == stored digit, also store, repeat once more
#     # if stored digits = len(3), do the same for num2 but digit == num1 triple digit and stored digits = len(2)
#     # if all tests passed, return 1 else 0
#     num1, num2 = list(str(num1)), list(str(num2))
#     number = None
#     for i in range(len(num1)-2):
#         if num1[i] == num1[i+1] == num1[i+2]:
#             number = num1[i]
#             break
#     if number:
#         for i in range(len(num2)-1):
#             if number == num2[i] == num2[i+1]:
#                 return 1
#     return 0


# print(triple_double(451999277, 41177722899)) # 1


# def encode(s):
#     bp = {'a': '1', 'e': '2', 'i': '3', 'o': '4', 'u': '5'}
#     slist = list(s)
#     for i, c in enumerate(slist):
#         if c in bp.keys():
#             slist[i] = bp[c]
#     return ''.join(slist)

# def decode(s):
#     bp = {'1': 'a', '2': 'e', '3': 'i', '4': 'o', '5': 'u'}
#     slist = list(s)
#     for i, c in enumerate(slist):
#         if c in bp.keys():
#             slist[i] = bp[c]
#     return ''.join(slist)

# print(encode("hi there"))
# print(decode("h3 th2r2"))


# def sort_array(source_array):
#     # Return a sorted array.
#     # in theory, need to know odds positions in original. create new odds only list.
#     odds = []
#     for num in source_array:
#         if num % 2 != 0:
#             odds.append(num)
#     odds.sort()
#     for i, num in enumerate(source_array):
#         if num % 2 != 0:
#             source_array[i] = odds.pop(0)
#     return source_array

# print(sort_array([5, 8, 6, 3, 4]))


# def add(n):
#     def adder(x):
#         return x + n
#     return adder

# print(add(5)(6))


# def solution(s):
#     final = ""
#     # each time the code sees an uppercase, need to input space before it
#     for c in s:
#         if c.isupper():
#             final += f" {c}"
#         else:
#             final += c
#     return final

# print(solution("camelCasing"))


# def meeting(s):
#     # makes this string uppercase
#     # gives it sorted in alphabetical order by last name.
#     # When the last names are the same, sort them by first name. Last name and first name of a guest come in the result between parentheses separated by a comma. (last, first)
#     u = s.upper()
#     # need to separate each first:last individually.
#     names = u.split(';') # this gives list of first:last
#     for i, name in enumerate(names):
#         names[i] = (name.split(':')[-1], name.split(':')[0])
#     return names

# print(meeting("Fred:Corwill;Wilfred:Corwill;Barney:Tornbull;Betty:Tornbull;Bjon:Tornbull;Raphael:Corwill;Alfred:Corwill"))


# # chatgpt
# def encrypt_this(text):
#     # Split the text into a list of words
#     words = text.split()

#     # Loop through each word and apply encryption rules
#     encrypted_words = []
#     for word in words:
#         # Convert first letter to ASCII code
#         first_letter_code = str(ord(word[0]))

#         # Switch second and last letter (if word has more than one letter)
#         if len(word) > 2:
#             second_letter = word[1]
#             last_letter = word[-1]
#             encrypted_letters = last_letter + word[2:-1] + second_letter
#         else:
#             encrypted_letters = word[1:]

#         # Add encrypted word to the list
#         encrypted_word = first_letter_code + encrypted_letters
#         encrypted_words.append(encrypted_word)

#     # Join encrypted words back into a string
#     encrypted_text = " ".join(encrypted_words)
#     return encrypted_text

# print(encrypt_this("hello world i am alex"))


# def find_uniq(arr):
#     # best case need to check 3 nums. as long as check 3 nums and 1 is repeat, return the other...
#     # create a set?
#     # can also sort. will need to sort. bc combinations could be limitless.
#     # after sort, check if index 0 = 1, if true then last value unique, else first value unique
#     arr.sort()
#     if arr[0] == arr[1]:
#         return arr[-1]
#     else:
#         return arr[0]

# print(find_uniq([ 2, 1, 1, 1, 1, 1 ]))


# def find_even_index(arr):
#     #your code here
#     # will for loop through each index, compare all numbers prev and after that index, excluding index and starting at index 0, left side is 0 (as well as last index right side = 0)
#     for i, n in enumerate(arr):
#         if sum(arr[:i]) == sum(arr[i+1:]):
#             return i
#     return -1

# print(find_even_index([20,10,-80,10,10,15,35]))


# def dig_pow(n, p):
#     # your code
#     # a bit confusing,
#     # find the sum of all digits to power of second number first, and check if entirely divisible by the first number, if so, return that number that it is divisible by
#     total = 0
#     for num in list(str(n)):
#         total += int(num) ** p
#         p += 1
#     return total // n if total % n == 0 else -1

# print(dig_pow(46288, 3))


# def solution(s):
#     # split a string in pairs. if odd, then last c paired w "_"
#     if len(s) % 2 != 0:
#         s += "_"
#     final = []
#     for i in range(0, len(s), 2):
#         final.append(s[i:i+2])
#     return final

# print(solution("asdfadsfg"))


# def is_pangram(s):
#     # grab alpha only and lowercase. how to check for all leters?
#     # can start by looping through
#     letters = list('abcdefghijklmnopqrstuvwxyz')
#     for c in s.lower():
#         if c in letters:
#             letters.remove(c)
#         if len(letters) == 0:
#             return True
#     return False

# print(is_pangram("The quick, brown fox jumps over the lazy dog!"))


# def adjacent_element_product(array):
#     # the other way is to run through entire list...which is needed since you need to check ALL pairs in order to determine you have the max..
#     max = None
#     for i in range(len(array)-1):
#         # take the first num and adjacent to the right
#         if max == None or array[i] * array[i+1] > max:
#             max = array[i] * array[i+1]
#     return max

# print(adjacent_element_product([-23, 4, -5, 99, -27, 329, -2, 7, -921]))


# def sum_of_n(n):
#     running = 0
#     lst = []
#     for i in range(abs(n)+1):
#         if n > 0:
#             running += i
#         else:
#             running -= i
#         lst.append(running)
#     return lst

# print(sum_of_n(3)) # [0,1,3,6]


# def shorter_reverse_longer(a,b):
#     # a + reverse(b) + a
#     if len(a) >= len(b):
#         return f'{b}{a[::-1]}{b}'
#     return f'{a}{b[::-1]}{a}'


# print(shorter_reverse_longer('abcabc', 'longerer'))


# def find_digit(num, nth):
#     #your code here
#     # grab absolute value of num
#     # if nth < 0 return -1
#     # if out of index, return 0
#     if nth <= 0:
#         return -1
#     elif nth > len(str(num)):
#         return 0
#     else:
#         return int(str(abs(num))[::-1][nth-1])

# print(find_digit(5673, 4))
# print('hello there')


# def vaporcode(s):
#     return '  '.join(list(''.join(s.upper().split(' '))))


# print(vaporcode("Why isn't my code working?"))


# def add(x):
#     def add_more(y):
#         print(add, add_more)
#         return x + y
#     print(add, add_more)
#     return add_more

# add_5 = add(5)
# print(add_5(1)) # Output: 15


# def multiply_all(lst):
#     return lambda multiplier: [n ** multiplier for n in lst]
#     # def multiply_by(multiplier):
#     #     return [num * multiplier for num in lst]
#     # return multiply_by

# print(multiply_all([1, 2, 3])(10))

# multiply_all_1 = multiply_all([1, 2, 3])
# print(multiply_all_1(10))

# def initial_num(x):
#     def whole_equation(y):
#         # print(x, y)
#         return x ** y
#     return whole_equation

# print(initial_num(2)(10))


# def blah(n):
#     return n ** 2 if n > 5 else n

# print(blah(5))
# print(blah(10))
# print(blah(15))


# def reverse_number(n):
#     # to reverse an int, need to turn into a str
#     # account for negative numbers
#     num = str(n)
#     x= '0'
#     if num[0] == '-':
#         num = num[1:]
#         x = '-'
#     return int(x + num[::-1])

# print(reverse_number(123))
# print(reverse_number(-456))
# print(reverse_number(1000))


# class Fighter(object):
#     def __init__(self, name, health, damage_per_attack):
#         self.name = name
#         self.health = health
#         self.damage_per_attack = damage_per_attack

#     def __str__(self): return "Fighter({}, {}, {})".format(self.name, self.health, self.damage_per_attack)
#     __repr__=__str__

# def declare_winner(fighter1, fighter2, first_attacker):
#     # first\_attacker makes first strike, take his attack to reduce other fighter's health
#     turn = fighter1 if fighter1.name == first_attacker else fighter2
#     while fighter1.health >= 0 and fighter2.health >= 0:
#         if turn == fighter1:
#             fighter2.health -= fighter1.damage_per_attack
#             turn = fighter2
#         else:
#             fighter1.health -= fighter2.damage_per_attack
#             turn = fighter1
#     print(fighter1, fighter2)
#     return fighter1.name if fighter2.health <= 0 else fighter2.name


# print(declare_winner(Fighter("susan", 20, 10),Fighter("Thomas", 5, 4), "Thomas"))

# def print_kwargs(**kwargs):
#     # for key, value in kwargs.items():
#     #     print(f"{key}: {value}")
#     return kwargs

# print(type(print_kwargs(a=1, b='two', c=True)))



# def sum_args(*args):
#     return sum(args)
#     # return sum(n for n in args)

# print(sum_args(2, 4, 5))

# def has_unique_chars(string):
#     return len(string) == len(set(string))


# def area_largest_square(r):
#     # radius is the distance from circle's center to its edge in a straight line
#     return 2*r**2

# print(area_largest_square(7))


# def my_languages(d):
#     # need to get k, v to match them
#     # final = []
#     # for key in d:
#     #     if d[key] >= 60:
#     #         final.append(key)
#     # return final
#     final = sorted(d, key=(lambda k: d[k]), reverse=True)
#     return list(filter(lambda lang: lang if d[lang] >= 60 else None, final))

# print(my_languages({"Java": 10, "Ruby": 80, "Python": 65}))

# def explode(s):
#     # new_string = ''
#     # for c in s:
#     #     new_string += int(c) * c
#     # return new_string
#     return ''.join(int(c) * c for c in s)

# print(explode('312'))

# def some_kwargs(**kwargs):
#     for k, v in kwargs.items():
#         print(f'{k}:', v)

# kwargs = {"kwarg_1": "Val", "kwarg_2": "Harper", "kwarg_3": "Remy"}
# some_kwargs(**kwargs)


# def multiply(*args, **kwargs):
#     z = 1
#     for num in args:
#         z *= num
#     print(z, type(kwargs))
#     # return (type(args))

# print(multiply(4, 5))
# print(multiply(10, 9))
# print(multiply(2, 3, 4))
# print(multiply(3, 5, 10, 6, john='smith'))


# def sum_dig_pow(a, b): # range(a, b + 1) will be studied by the function
#     # your code here
#     final = []
#     for n in range(a, b+1):
#         # for each number, if number  more than 1 digit, separate digits, raise each consecutive digit to power of 1, 2, 3, 4, etc. return sum == number
#         # now separate number
#         total = 0
#         for i in range(len(str(n))):
#             # this takes '89' and loops through each digit
#             total += int(str(n)[i]) ** (i + 1) # this takes each digit to power of index starting at 1 and adds to total sum for that number
#             # check if total = number
#         if total == n:
#             final.append(n)
#     return final

# print(sum_dig_pow(1,10))
# print(sum_dig_pow(1,100))
# print(sum_dig_pow(10,89))
# print(sum_dig_pow(10,100))
# print(sum_dig_pow(90,100))

# def sum_consecutives(s):
#     #good luck
#     # for loop. only sum curr num if next num is the same, if different num, add either the num or sum to the list
#     # only append to new list when numbers different, if not, keep adding
#     sum_nums = 0
#     final = []
#     for i in range(len(s)-1):
#         # if number is diff than next and no sum nums, add num
#         if s[i] != s[i+1] and not sum_nums:
#             final.append(s[i])
#         # else if num diff than next but yes sum nums, add num + sumnums
#         elif s[i] != s[i+1]:
#             final.append(sum_nums + s[i])
#         # else if num same than next, sumnums += num
#         else:
#             sum_nums += s[i]
#     return final

# print(sum_consecutives([1,4,4,4,0,4,3,3,1]))
# print(sum_consecutives([1,4,4,4,0,4,3,3,1,1]))

# # matrixAddition(
# #   [ [1, 2, 3],
# #     [3, 2, 1],
# #     [1, 1, 1] ],
# # //      +
# #   [ [2, 2, 1],
# #     [3, 2, 3],
# #     [1, 1, 3] ] )

# # // returns:
# #   [ [3, 4, 4],
# #     [6, 4, 4],
# #     [2, 2, 4] ]

# def matrix_addition(a, b):
#     # your code here
#     final = []
#     for i, array in enumerate(a):
#         chunks = []
#         for j, num in enumerate(array):
#             chunks.append(a[i][j] + b[i][j])
#         final.append(chunks)
#     return final

# print(matrix_addition(
#     [ [1, 2, 3],
#     [3, 2, 1],
#     [1, 1, 1] ],
#     [ [2, 2, 1],
#     [3, 2, 3],
#     [1, 1, 3] ]))

# Pair of gloves
# Winter is coming, you must prepare your ski holidays. The objective of this kata is to determine the number of pair of gloves you can constitute from the gloves you have in your drawer.

# Given an array describing the color of each glove, return the number of pairs you can constitute, assuming that only gloves of the same color can form pairs.

# Examples:
# input = ["red", "green", "red", "blue", "blue"]
# result = 2 (1 red pair + 1 blue pair)

# input = ["red", "red", "red", "red", "red", "red"]
# result = 3 (3 red pairs)

# def number_of_pairs(gloves):
#     glove_pairs = {}
#     for glove in gloves:
#         glove_pairs[glove] = glove_pairs.get(glove, 0) + 1
#     pairs = 0
#     for value in glove_pairs.values():
#         pairs += value // 2
#     return pairs

# print(number_of_pairs(["red", "green", "red", "blue", "blue"]))
# print(number_of_pairs(["red", "red", "red", "red", "red", "red"]))

# def calc(expr):
#     # need for loop through string..if can be int or float, convert, if operand, need to change its place to 1 index position to its left, if consecutive numbers wo operands, use next operand for all of them
#     # should i change the string first to have operands in right place?
#     # can create a list with the numbers if i and i+1 are both numbers to store them, always check i and i+1..if i and i+1 are numbers, store in list? if not, the perform operation
#     return eval(expr)

# print(calc("5 - 1 + 2 + 4 * + 3"))

# def clean_string(s):
#     # a backspace can be one or more and deletes characters that come before it
#     # basically pop() should delete the last from string
#     backspaced = []
#     for c in s:
#         if c == '#':
#             if not backspaced:
#                 continue
#             else:
#                 backspaced.pop()
#         else:
#             backspaced.append(c)
#     return ''.join(backspaced)

# print(clean_string('abc#d##c'))
# print(clean_string('abc##d######'))


# class Solution:
#     def containsDuplicate(self, nums: List[int]) -> bool:
# #         could loop through each elem
# #         easy solution:
#         for num in nums:
#             if nums.count(num) > 1:
#                 return True
#         return False


# def solve(string):
#     # check if char is consonant, then get its value and store it in dict
#     cons = set('bcdfghjklmnpqrstvwxyz')
#     values = {}
#     for c in string:
#         if c in cons:
#             values[c] = ord(c) - 96 # 'a' gives 1
#     # now i have dict of all leters in string with their values..need to now store/compare consecutive consonants in string
#     # can do a running sum of values IF still consonants,
#     is_c_cons = False
#     tally = 0
#     for i, c in enumerate(string):
#         if c in cons:
#             is_c_cons = True
#         else:
#             is_c_cons = False
#         if is_c_cons:
#             tally += values[c]

# print(solve('strength'))


# def highest_rank(arr):
#     # return the number w the highest count within the array
#     # if tie, return the largest number
#     count = 0
#     maximum = None
#     for num in set(arr):
#         this_num_count = arr.count(num)
#         if this_num_count > count:
#             count = this_num_count
#             maximum = num
#         elif this_num_count == count:
#             if num > maximum:
#                 maximum = num
#     return maximum

# print(highest_rank([12, 10, 8, 12, 7, 6, 4, 10, 12]))
# print(highest_rank([12, 10, 8, 12, 7, 6, 4, 10, 12, 10]))
# print(highest_rank([12, 10, 8, 8, 3, 3, 3, 3, 2, 4, 10, 12, 10]))


# def pyramid(n):
# # input is integer
# # output is array of arrays
# # cannot be less than 0, if 0 then empty array
# # each array should contain 1 as int inside each index position
# # create initial empty list
# # while loop until 0 reached? insert each new arr at index 0
# # in while loop, create single arr with n length of 1 values
# # after each loop, reduce n by 1
#     final = []
#     while n > 0:
#         # arr = list('1'*n) # need to create [1, 1, 1]
#         arr = [1 for _ in range(n)]
#         final.insert(0, arr)
#         n -= 1
#     # print(10*['i'])
#     return final

# print(pyramid(5))


# # Write a class that, when given a string, will return an uppercase string with each letter shifted forward in the alphabet by however many spots the cipher was initialized to.

# # For example:

# # c = CaesarCipher(5); # creates a CipherHelper with a shift of five
# # c.encode('Codewars') # returns 'HTIJBFWX'
# # c.decode('BFKKQJX') # returns 'WAFFLES'
# # If something in the string is not in the alphabet (e.g. punctuation, spaces), simply leave it as is.
# # The shift will always be in range of [1, 26].

# class CaesarCipher(object):
#     # need to lowercase the string, then for each alpha character move forward if encode or backward if decode
#     # range in shift will be within 1, 26
#     # need formula to do if c after shift is < 1 or > 26, then subtract accordingly or something
#     def __init__(self, shift):
#         self.shift = shift

#     def encode(self, string):
#         # returns + shift
#         # will change the string in-place not out of place
#         st = list(string.lower())
#         for i, c in enumerate(st):
#             if c.isalpha():
#                 # change c by + shift, if out of range, do result - 26 or something
#                 new = ord(c) + self.shift # NEED TO GET THE INT OF C HERE
#                 # if letter goes over 'z', subtract 26 from ord value
#                 if new > 122:
#                     st[i] = chr(new - 26)
#                 else:
#                     st[i] = chr(new)
#         return ''.join(st).upper()

#         # 97  a
#         # 122 z

#     def decode(self, string):
#         # returns - shift
#         st = list(string.lower())
#         for i, c in enumerate(st):
#             if c.isalpha():
#                 # change c by + shift, if out of range, do result - 26 or something
#                 new = ord(c) - self.shift # NEED TO GET THE INT OF C HERE
#                 # if letter goes over 'z', subtract 26 from ord value
#                 if new < 97:
#                     st[i] = chr(new + 26)
#                 else:
#                     st[i] = chr(new)
#         return ''.join(st).upper()

# c = CaesarCipher(5)
# print(c.encode('Code.wars'))
# print(c.decode('BFKKQJX'))


# Your Task
# You will be given a wishlist (array), containing all possible items. Each item is in the format: {name: "toy car", size: "medium", clatters: "a bit", weight: "medium"} (Ruby version has an analog hash structure, see example below)

# You also get a list of presents (array), you see under the christmas tree, which have the following format each: {size: "small", clatters: "no", weight: "light"}

# Your task is to return the names of all wishlisted presents that you might have gotten.

# Rules
# Possible values for size: "small", "medium", "large"
# Possible values for clatters: "no", "a bit", "yes"
# Possible values for weight: "light", "medium", "heavy"
# Don't add any item more than once to the result
# The order of names in the output doesn't matter
# It's possible, that multiple items from your wish list have the same attribute values. If they match the attributes of one of the presents, add all of them.
# Example
# wishlist = [{'name': "mini puzzle", 'size': "small", 'clatters': "yes", 'weight': "light"},
#             {'name': "toy car", 'size': "medium", 'clatters': "a bit", 'weight': "medium"},
#             {'name': "card game", 'size': "small", 'clatters': "no", 'weight': "light"}]
# presents = [{'size': "medium", 'clatters': "a bit", 'weight': "medium"},
#             {'size': "small", 'clatters': "yes", 'weight': "light"}]
# # guess_gifts(wishlist, presents) # => must return ["Toy Car", "Mini Puzzle"]

# def guess_gifts(wishlist, presents):
#     # input: 2 lists, each with multiple dictionaries with string attr
#     # output: list with string attr
#     # check for every present, if all of its 3 attributes match w 1 or multiple items in wishlist
#     possible = []
#     for i, present in enumerate(presents):
#         # print(i, present['weight'])
#         for j, wishlist_p in enumerate(wishlist):
#             if present['weight'] == wishlist_p['weight'] and present['size'] == wishlist_p['size'] and present['clatters'] == wishlist_p['clatters']:
#                 possible.append(wishlist_p['name'])
#     return list(set(possible))

# print(guess_gifts(wishlist, presents))


# def zero(): #your code here
# def one(): #your code here
# def two(): #your code here
# def three(): #your code here
# def four(): #your code here
# def five(): #your code here
# def six(): #your code here
# def seven(): #your code here
# def eight(): #your code here
# def nine(): #your code here

# def plus(): #your code here
# def minus(): #your code here
# def times(): #your code here
# def divided_by(): #your code here


# print(seven(times(five()))) # must return 35
# print(four(plus(nine()))) # must return 13
# print(eight(minus(three()))) # must return 5
# print(six(divided_by(two()))) # must return 3


# Write a function, which takes a non-negative integer (seconds) as input and returns the time in a human-readable format (HH:MM:SS)

# HH = hours, padded to 2 digits, range: 00 - 99
# MM = minutes, padded to 2 digits, range: 00 - 59
# SS = seconds, padded to 2 digits, range: 00 - 59
# The maximum time never exceeds 359999 (99:59:59)

# You can find some examples in the test fixtures.

# def make_readable(seconds):
#     # input -> int
#     # output -> str w hh:mm:ss as numbers
#     # need to figure out how each s, m, h will display its number accordingly...
#     # can take a number and retain its remainder converted to s, then m, then h...
#     # need to first check its remainder as divided by 60, then its remainder by 3600? that would give you the resting minutes in seconds, so would need to convert those seconds to minute notation probably? same with hours?
#     # if divide 21 / 10, you get 1 as remainder, 25 / 10 you get 5. its all in its base notation

#     # logically, I would split the number into chunks
#     # last chunk div by 60
#     secs = '0' + str(seconds % 60) if seconds % 60 < 10 else str(seconds % 60) # gets seconds
#     mins = '0' + str((seconds % (60*60)) // 60 ) if ((seconds % (60*60)) // 60 ) < 10 else str((seconds % (60*60)) // 60 ) # need minutes
#     hrs = '0' + str((seconds // (60*60))) if ((seconds // (60*60))) < 10 else str((seconds // (60*60))) # gets hours
#     return f'{hrs}:{mins}:{secs}'

# print(make_readable(0))
# print(make_readable(5))
# print(make_readable(60))
# print(make_readable(86399))
# print(make_readable(359999))

# def narcissistic(value):
#     exp = len(str(value))
#     numbers = [int(d)**exp for d in (str(value))]
#     return sum(numbers) == value

# print(narcissistic(153))



# def order(sentence):
#     # code here
# #   need to split the str, then check if there is an integer in string, and place that word or insert in its right position in new array
#     def int_order(word):
#         # list_sentence = sentence.split()
#         # for word in (list_sentence):
#         for c in word:
#             if c.isnumeric():
#                 # need to modify original list...so pop word at its index, and insert that popped word at new index...
#                 return int(c)

#     return ' '.join(sorted(sentence.split(), key=int_order))

# print(order('4of Fo1r pe6ople g3ood th5e the2'))


# def sort_by_length(arr):
#     # need to sort array from small to largest in length...
#     return sorted(arr, key=len)

# print(sort_by_length(["Telescopes", "Glasses", "Eyes", "Monocles"]))

# def remove_url_anchor(url):

#     # TODO: complete
#     # need to look at url, see 1st instance of '#' symbol, and only return whatever is before that symbol
#     return url.partition('#')

# print(remove_url_anchor("www.codewars.com#about#me"))
# print(remove_url_anchor("www.codewars.com?page=1"))


# write the function is_anagram
# def is_anagram(test, original):
#     # so if we can compare letters by re-arraning w a fn, should solve problem, taking sentence case into acct
#     return sorted(test.lower()) == sorted(original.lower())

# print(is_anagram("foefeT", "toffee"))


# def sequence_sum(begin_number, end_number, step):
#     # input three numbers as integers
#     # output one integer
#     total = 0
#     for num in range(begin_number, end_number+1, step):
#         total += num
#     return total

# print(sequence_sum(1,5,1))

# def order_weight(string):
#     # input: string of numbers separated by spaces
#     # output: string of numbers separated by spaces
#     # need to separate strings by whitespace, change each string to integer for comparison, sort smallest to largest integer, if same integer sort by their string form, when finished sorting, return string of joined integers
#     str_list = string.split(' ')
#     sums_list = []
#     for weight in str_list:
#         dsum = 0
#         for d in weight:
#             dsum += int(d)
#         sums_list.append(dsum)
#     return (sums_list)
    # return sorted(int_list)

# def order_weight(_str):
#     return ' '.join(sorted((_str.split(' ')), key=lambda x: sum([int(c) for c in x])))
#     # return (int(c) for c in _str.split())
#     # return sum([1,2,3,4,5])
#     # return sum(1,2,3,4,5)

# print(order_weight("56 65 74 100 99 68 86 180 90")) # "100 180 90 56 65 74 68 86 99"


# A single die can only be counted once in each roll. For example, a given "5" can only count as part of a triplet (contributing to the 500 points) or as a single 50 points, but not both in the same roll.

# Example scoring
#  Throw       Score
#  ---------   ------------------
#  5 1 3 4 1   250:  50 (for the 5) + 2 * 100 (for the 1s)
#  1 1 1 3 1   1100: 1000 (for three 1s) + 100 (for the other 1)
#  2 4 4 5 4   450:  400 (for three 4s) + 50 (for the 5)

# def score(dice):
#     # input: array of 5 numbers from 1-6 random
#     # output: integer of total points
#     # if a number is contained exactly 3 times in array, sum those special points
#     # elif for every 1 or 5 in array, sum 100 or 50 points
#     # esta mucho mas complicado de lo que pensaba...
#     # me la estoy complicando siento
#     points = 0
#     dnums = {}
#     for num in set(dice):
#         dnums[num] = dice.count(num)
#         if dnums[num] >= 3:
#             if num == 1:
#                 points += num * 1000
#                 dnums[num] -= 3
#                 points += dnums[num] * 100
#             elif num == 5:
#                 points += num * 100
#                 dnums[num] -= 3
#                 points += dnums[num] * 100
#             else:
#                 points += num * 100
#         else:
#             if num == 1:
#                 points += dnums[num] * 100
#             elif num == 5:
#                 points += dnums[num] * 50
#     return points

# print(score([5,1,3,4,1]))
# print(score([1,1,1,1,1]))
# print(score([1,1,1,1,3]))
# print(score([2,4,4,5,4]))


# def calc():
#     # there is a pattern here: 2^n always follows 2, 4, 8, 6, 2, 4, 8, 6, etc
#     for num in range(12, 16):
#         for exp in range(1, 16):
#             print(num, 'to the', exp, f"Result is {num**exp}")
#     return pow(3, 3, 7) % 7

# print(calc())


# def strings(word1, word2):
#     return ''.join(word1) == ''.join(word2)


# print(strings(['ab', 'c'], ['a', 'bc']))


# def steps(num):
#     steps = 0
#     while num > 0:
#         if num % 2 == 0:
#             num /= 2
#         else:
#             num -= 1
#         steps += 1
#     return steps

# print(steps(4))


# def richest(xlist):
#     for i, customer in enumerate(xlist):
#         xlist[i] = sum(customer)
#     return max(xlist)

# print(richest([[7,1,3],[2,8,7],[1,9,5]]))


# def running_sum(nums):
#     # nums is a list
#     count = 0
#     running_list = []
#     for num in nums:
#         count += num
#         running_list.append(count)
#     return running_list

# print(running_sum([3,1,2,10,1]))


# # You need to write regex that will validate a password to make sure it meets the following criteria:

# # At least six characters long
# # contains a lowercase letter
# # contains an uppercase letter
# # contains a digit
# # only contains alphanumeric characters (note that '_' is not alphanumeric)

# import re
# # pattern = re.compile(r'')
# # creo que ya tengo todo lo que necesito para cumplir..

# test = re.search(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?!.*_)(?!.*\W).{6,}', '2Aabcdf')
# print(test)
# # test2 = re.search(r'\w{9}', '2Aabcdf__\t')
# # print(test2)


# def done_or_not(board):
#     for i in range(0, 9):
#       (l, c, b) = (set(), set(), set())
#       for j in range(0, 9):
#           c.add(board[i][j])
#           l.add(board[j][i])
#           b.add(board[0 + j // 3][((i * 3) % 9) + j % 3])
#       if len(l)!=9 or len(c)!=9 or len(b)!=9:
#           return 'Try again!'
#     return "Finished!"


# print(done_or_not([  [1, 3, 2, 5, 7, 9, 4, 6, 8]
#                         ,[4, 9, 8, 2, 6, 1, 3, 7, 5]
#                         ,[7, 5, 6, 3, 8, 4, 2, 1, 9]
#                         ,[6, 4, 3, 1, 5, 8, 7, 9, 2]
#                         ,[5, 2, 1, 7, 9, 3, 8, 4, 6]
#                         ,[9, 8, 7, 4, 2, 6, 5, 3, 1]
#                         ,[2, 1, 4, 9, 3, 5, 6, 8, 7]
#                         ,[3, 6, 5, 8, 1, 7, 9, 2, 4]
#                         ,[8, 7, 9, 6, 4, 2, 1, 5, 3]]))

# def done_or_not(board): #board[i][j]
#     # okay. rows are easy just create a set for each row to check for duplicates..
#     # n is a number
#     # if n at position 1, n cannot be in that row, nor that column, nor in the range of variable 3x3 grid
#     # we could create a set for each row, each column, and each 3x3. that would check if all are unique values...
#     # row set...for each list, check if set(list) has length of 9 to continue
#     for i, row in enumerate(board):
#         if len(set(row)) != 9:
#             return 'Try again!'
#     # col set...for each row at row[index], check if num not in set to add to set and continue
#     for i in range(len(board)):
#         cols = set()
#         for j in range(len(board)):
#             if board[j][i] in cols:
#                 return 'Try again!'
#             cols.add(board[j][i])
#             # print(cols)
#     # 3x3 set...for each row, if row[index] in range 0:3,3:6,6:9, check row numbers in that range, and two above/below rows as well at same range, check if num not in 3x3 set to add to set and continue
#     grid = set()
#     for i in range(len(board)):
#         for j in range(len(board)):
#             if i in range(0,3):
#                 if j in range(0,3):
#                     if board[i][j] in grid:
#                         return 'Try again!'
#                     grid.add(board[i][j])
#                 elif j in range(3,6):
#                     if board[i][j] in grid:
#                         return 'Try again!'
#                     grid.add(board[i][j])
#                 elif j in range(6,9):
#                     if board[i][j] in grid:
#                         return 'Try again!'
#                     grid.add(board[i][j])
#             grid = set()
#             if i in range(3,6):
#                 if j in range(0,3):
#                     if board[i][j] in grid:
#                         return 'Try again!'
#                     grid.add(board[i][j])
#                 elif j in range(3,6):
#                     if board[i][j] in grid:
#                         return 'Try again!'
#                     grid.add(board[i][j])
#                 elif j in range(6,9):
#                     if board[i][j] in grid:
#                         return 'Try again!'
#                     grid.add(board[i][j])
#             grid = set()
#             if i in range(6,9):
#                 if j in range(0,3):
#                     if board[i][j] in grid:
#                         return 'Try again!'
#                     grid.add(board[i][j])
#                 elif j in range(3,6):
#                     if board[i][j] in grid:
#                         return 'Try again!'
#                     grid.add(board[i][j])
#                 elif j in range(6,9):
#                     if board[i][j] in grid:
#                         return 'Try again!'
#                     grid.add(board[i][j])

#     return 'Finished!'

# print(done_or_not([  [1, 3, 2, 5, 7, 9, 4, 6, 8]
#                         ,[4, 9, 8, 2, 6, 1, 3, 7, 5]
#                         ,[7, 5, 6, 3, 8, 4, 2, 1, 9]
#                         ,[6, 4, 3, 1, 5, 8, 7, 9, 2]
#                         ,[5, 2, 1, 7, 9, 3, 8, 4, 6]
#                         ,[9, 8, 7, 4, 2, 6, 5, 3, 1]
#                         ,[2, 1, 4, 9, 3, 5, 6, 8, 7]
#                         ,[3, 6, 5, 8, 1, 7, 9, 2, 4]
#                         ,[8, 7, 9, 6, 4, 2, 1, 5, 3]]))

# Your job is to write a function which increments a string, to create a new string.

# If the string already ends with a number, the number should be incremented by 1.
# If the string does not end with a number. the number 1 should be appended to the new string.
# Examples:

# foo -> foo1

# foobar23 -> foobar24

# foo0042 -> foo0043

# foo9 -> foo10

# foo099 -> foo100

# Attention: If the number has leading zeros the amount of digits should be considered.


# def increment_string(string):
#     # okay, so take the string, iterate to check if c is letter, add to letters list, number to numbers list
#     # once you have a numbers list, join numbers, add + 1 to that number (but what if 0's before?)
#         # leading 0s Not Permitted. So will have to add leading 0s to strings list or split from numbers list...could do a count to check for 0s, if 0 after letter, change count to 1, if no longer 0, change count to 2
#     # NUMBERS COULD BE BETWEEN LETTERS, ONLY COUNT CONSECUTIVE LAST NUMBERS
#     # IF 099, THEN + 1 WOULD BE 100 NOT 0100
#     # ALSO, 000 SHOULD EQUAL 001 NOT 0001
#     count = 0
#     letters = ''
#     numbers = ''
#     for i, c in enumerate(string[::-1]):
#         if c.isnumeric() and count == 0:
#             numbers += c
#         else:
#             count = 1
#             letters += c
#     # need to check if numbers starts with 0s,
#     letters = letters[::-1]
#     numbers = numbers[::-1]
#     if not numbers:
#         numbers = '1'
#     else:
#         plus_one = str(int(numbers) + 1) # this removes leading 0s
#         if len(numbers) <= len(plus_one):
#             numbers = plus_one
#         else:
#             numbers = numbers[0:(len(numbers)-len(plus_one))] + plus_one
#     return ''.join(letters + numbers)

# print(increment_string('foo'))
# print(increment_string('foobar23'))
# print(increment_string('foo0042'))
# print(increment_string('fo99o099'))
# print(increment_string('fo99o0099'))
# print(increment_string('fo99o009899'))


# def scramble(s1, s2):
#     # do a dict approach
#     chars = {}
#     for c in set(s1):
#         chars[c] = s1.count(c)
#     for c in s2:
#         if c in chars:
#             chars[c] -= 1
#         else:
#             return False
#         if chars[c] < 0:
#             return False
#     return True

# print(scramble('cedewaraaossoqqyt', 'codewars'))


# def zeros(n):
#     # factorial:
#     # need to multiply successive integers up to given number input
#     # zeros(3) = 1 * 2 * 3 = 6 -> zero 0s
#     # order doesn't matter, can go from up to down
#     # start with n, then go down n-1 until that == 1
#     # need to multiply each n * n-1...
#     if n == 1:
#         print('n is at end: 1')
#         return prod
#     else:
#         print(f'n is {n}')
#         prod = n * zeros(n-1)

# print(zeros(6))


# def first_non_repeating_letter(string):
#     # logic: check the entire string for chars, count them, look for first non-repeating
#     # lower same as upper, but return accordingly
#     # i'm thinking a for loop to create a dict of chars, then another for loop afterwards to check dict items vs non-dict
#     # chars = {}
#     # for c in string:
#     #     if c.lower() in chars:
#     #         chars[c.lower()] = chars[c.lower()] + 1
#     #     else:
#     #         chars[c.lower()] = 1

#     # for c in string:
#     #     if chars[c.lower()] == 1:
#     #         return c
#     # return ''
#     for c in string:
#         print(string.index(c))

# print(first_non_repeating_letter([1, 2, 3, 2, 4, 2, 5, 2]))
# print(first_non_repeating_letter('sTreSS'))

# Write a function that when given a URL as a string, parses out just the domain name and returns it as a string. For example:

# * url = "http://github.com/carbonfive/raygun" -> domain name = "github"
# * url = "http://www.zombie-bites.com"         -> domain name = "zombie-bites"
# * url = "https://www.cnet.com"                -> domain name = cnet"


# def domain_name(string):
    # need to check if '.' on either side to indicate start/finish of domain name
    # need to check if before '.' there is http:// or www
    #  can check if string before is '/' then take whatever is after until there is a '.'
    # domain = ''
    # if 'www' in string:
    #     # check where first '.' is, return whatever comes after
    #     dots = 0
    #     for c in string:
    #         if c == '.' and dots == 0:
    #             dots += 1
    #             continue
    #         elif c == '.' and dots != 0:
    #             break
    #         if dots == 1:
    #             domain += c
    # elif 'http' in string:
    #     http = ''
    #     for c in string:
    #         if c in {':', '/'}:
    #             http += c
    #             continue
    #         elif c == '.':
    #             break
    #         if http == '://':
    #             domain += c
    # else:
    #     for c in string:
    #         if c == '.':
    #             break
    #         else:
    #             domain += c
    # return domain
#     return string.split('www.')


# print(domain_name("https://youtube.com"))
# print(domain_name("https://www.you-tube.com"))
# print(domain_name("www.you-tube.com"))

# ROT13 is a simple letter substitution cipher that replaces a letter with the letter 13 letters after it in the alphabet. ROT13 is an example of the Caesar cipher.

# Create a function that takes a string and returns the string ciphered with Rot13. If there are numbers or special characters included in the string, they should be returned as they are. Only letters from the latin/english alphabet should be shifted, like in the original Rot13 "implementation".

# def rot13(message):
#     # need to loop through each
#     rot = ''
#     for c in message:
#         # if c is not a letter, add it to rot
#         if not c.isalpha():
#             rot += c
#         # if c is a letter, add 13 in alphabet
#         if c.isalpha():
#             # if c is in capital range
#             if ord(c) in range(65, 90 + 1):
#                 # if c less than 'N' do c + 13, else -13
#                 if ord(c) < 78:
#                     rot += chr(ord(c) + 13)
#                 else:
#                     rot += chr(ord(c) - 13)
#             # if c is in lowercase range
#             elif ord(c) in range(ord('a'), ord('z') + 1):
#                 # if c less than 'n' do c + 13, else -13
#                 if ord(c) < ord('n'):
#                     rot += chr(ord(c) + 13)
#                 else:
#                     rot += chr(ord(c) - 13)

#     return rot


# print(rot13('aA bB zZ 1234 *!?%'))


# 65 A
# 77 M
# 78 N
# 90 Z

# 97 a
# 109 m
# 110 n
# 122 z

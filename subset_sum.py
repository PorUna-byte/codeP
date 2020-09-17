# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
def merge_list(li, x):
    i = j = 0
    ans = []
    while i < len(li) and j < len(li):
        if li[i] < li[j] + x:
            ans.append(li[i])
            i = i + 1
        elif li[i] > li[j] + x:
            ans.append(li[j] + x)
            j = j + 1
        else:
            ans.append(li[i])
            i = i+1
            j = j+1
    while i < len(li):
        ans.append(li[i])
        i = i+1
    while j < len(li):
        ans.append(li[j]+x)
        j = j+1
    return ans
def exact_subset_sum(s,t):
    n = len(s)
    l = []
    l.append([0])
    for i in range(n):
        l.append(merge_list(l[i], s[i]))
        j = 0
        while j < len(l[i+1]):
            if(l[i+1][j] > t):
                l[i+1].remove(l[i+1][j])
                j = j-1
            j = j+1
    return l[n].pop()
def trim(l,deta):
    m = len(l)
    l_ = []
    l_.append(l[0])
    last = l[0]
    i = 1
    while i < m:
        if l[i] > last*(1+deta):
            l_.append(l[i])
            last = l[i]
        i = i+1
    return l_
def approx_subset_sum(s,t,ep):
    n = len(s)
    l = []
    l.append([0])
    for i in range(n):
        temp = merge_list(l[i], s[i])
        temp = trim(temp, ep/(2*n))
        l.append(temp)
        j = 0
        while j < len(l[i+1]):
            if(l[i+1][j] > t):
                l[i+1].remove(l[i+1][j])
                j = j-1
            j = j+1
    return l[n].pop()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    li_ = [104, 102, 201, 101]
    li_.sort()
    ans_ = approx_subset_sum(li_, 308, 0.4)
    li = [10, 11, 12, 15, 20, 21, 22, 23, 24, 29]
    ans =trim(li, 0.1)
    print ans_
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

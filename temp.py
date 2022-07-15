import os
stri = "88,91,33,112,45,11,44,22"
print(stri)
s = stri.split(',')
for i in range(len(s)):
    s[i] = int(s[i])

list = []
list.append(min(s[0], s[6]))
list.append(max(s[1], s[3]))
list.append(max(s[2], s[4]))
list.append(min(s[5], s[7]))
for i in range(len(list)):
    list[i] = str(list[i])
print(list)


stri = ','.join(list)
print(stri)
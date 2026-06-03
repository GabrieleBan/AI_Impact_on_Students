#  es 




for i in range(1,100000000):
    divides=True
    for n in range(1,21):
        if not i%n ==0:
            divides=False
            break
    if divides:
        print(i)
        break

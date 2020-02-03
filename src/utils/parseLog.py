import numpy

if __name__ == '__main__':
    times = []
    seconds = 0
    with open('log.txt') as log:
        for line in log:
            seconds_chapter, time = line.split(' ')
            times.append(int(time))
            seconds += int(seconds_chapter)

    print(seconds, numpy.sum(times))
    print(seconds * 6 / numpy.sum(times))

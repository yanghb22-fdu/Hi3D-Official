#!/usr/bin/env python

import getopt
import math
import sys
import typing

import numpy
import PIL
import PIL.Image
import torch

from . import softsplat  # the custom softmax splatting layer

try:
    from .correlation import correlation  # the custom cost volume layer
except:
    sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# end

##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'lf'
arguments_strOne = './images/one.png'
arguments_strTwo = './images/two.png'
arguments_strVideo = './videos/car-turn.mp4'
arguments_strOut = './out.png'
arguments_strVideo2 = ''

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
    if strOption == '--one' and strArgument != '': arguments_strOne = strArgument # path to the first frame
    if strOption == '--two' and strArgument != '': arguments_strTwo = strArgument # path to the second frame
    if strOption == '--video' and strArgument != '': arguments_strVideo = strArgument # path to a video
    if strOption == '--video2' and strArgument != '': arguments_strVideo2 = strArgument # path to a video
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

def read_flo(strFile):
    with open(strFile, 'rb') as objFile:
        strFlow = objFile.read()
    # end

    assert(numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=1, offset=0) == 202021.25)

    intWidth = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=4)[0]
    intHeight = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=8)[0]

    return numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=intHeight * intWidth * 2, offset=12).reshape(intHeight, intWidth, 2)
# end

##########################################################

backwarp_tenGrid = {}

def backwarp(tenIn, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[3], dtype=tenFlow.dtype, device=tenFlow.device).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[2], dtype=tenFlow.dtype, device=tenFlow.device).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()
    # end

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenIn, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
# end

##########################################################

class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netFirst = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSecond = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThird = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFourth = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFifth = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSixth = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
            # end

            def forward(self, tenInput):
                tenFirst = self.netFirst(tenInput)
                tenSecond = self.netSecond(tenFirst)
                tenThird = self.netThird(tenSecond)
                tenFourth = self.netFourth(tenThird)
                tenFifth = self.netFifth(tenFourth)
                tenSixth = self.netSixth(tenFifth)

                return [tenFirst, tenSecond, tenThird, tenFourth, tenFifth, tenSixth]
            # end
        # end

        class Decoder(torch.nn.Module):
            def __init__(self, intChannels):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intChannels, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
                )
            # end

            def forward(self, tenOne, tenTwo, objPrevious):
                intWidth = tenOne.shape[3] and tenTwo.shape[3]
                intHeight = tenOne.shape[2] and tenTwo.shape[2]

                tenMain = None

                if objPrevious is None:
                    tenVolume = correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo)

                    tenMain = torch.cat([tenOne, tenVolume], 1)

                elif objPrevious is not None:
                    tenForward = torch.nn.functional.interpolate(input=objPrevious['tenForward'], size=(intHeight, intWidth), mode='bilinear', align_corners=False) / float(objPrevious['tenForward'].shape[3]) * float(intWidth)

                    tenVolume = correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=backwarp(tenTwo, tenForward))

                    tenMain = torch.cat([tenOne, tenVolume, tenForward], 1)

                # end

                return {
                    'tenForward': self.netMain(tenMain)
                }
            # end
        # end

        self.netExtractor = Extractor()

        self.netFirst = Decoder(16 + 81 + 2)
        self.netSecond = Decoder(32 + 81 + 2)
        self.netThird = Decoder(64 + 81 + 2)
        self.netFourth = Decoder(96 + 81 + 2)
        self.netFifth = Decoder(128 + 81 + 2)
        self.netSixth = Decoder(192 + 81)
    # end

    def forward(self, tenOne, tenTwo):
        intWidth = tenOne.shape[3] and tenTwo.shape[3]
        intHeight = tenOne.shape[2] and tenTwo.shape[2]

        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)

        objForward = None
        objBackward = None

        objForward = self.netSixth(tenOne[-1], tenTwo[-1], objForward)
        objBackward = self.netSixth(tenTwo[-1], tenOne[-1], objBackward)

        objForward = self.netFifth(tenOne[-2], tenTwo[-2], objForward)
        objBackward = self.netFifth(tenTwo[-2], tenOne[-2], objBackward)

        objForward = self.netFourth(tenOne[-3], tenTwo[-3], objForward)
        objBackward = self.netFourth(tenTwo[-3], tenOne[-3], objBackward)

        objForward = self.netThird(tenOne[-4], tenTwo[-4], objForward)
        objBackward = self.netThird(tenTwo[-4], tenOne[-4], objBackward)

        objForward = self.netSecond(tenOne[-5], tenTwo[-5], objForward)
        objBackward = self.netSecond(tenTwo[-5], tenOne[-5], objBackward)

        objForward = self.netFirst(tenOne[-6], tenTwo[-6], objForward)
        objBackward = self.netFirst(tenTwo[-6], tenOne[-6], objBackward)

        return {
            'tenForward': torch.nn.functional.interpolate(input=objForward['tenForward'], size=(intHeight, intWidth), mode='bilinear', align_corners=False) * (float(intWidth) / float(objForward['tenForward'].shape[3])),
            'tenBackward': torch.nn.functional.interpolate(input=objBackward['tenForward'], size=(intHeight, intWidth), mode='bilinear', align_corners=False) * (float(intWidth) / float(objBackward['tenForward'].shape[3]))
        }
    # end
# end

##########################################################

class Synthesis(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Basic(torch.nn.Module):
            def __init__(self, strType, intChannels, boolSkip):
                super().__init__()

                if strType == 'relu-conv-relu-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )

                elif strType == 'conv-relu-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )

                # end

                self.boolSkip = boolSkip

                if boolSkip == True:
                    if intChannels[0] == intChannels[2]:
                        self.netShortcut = None

                    elif intChannels[0] != intChannels[2]:
                        self.netShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0, bias=False)

                    # end
                # end
            # end

            def forward(self, tenInput):
                if self.boolSkip == False:
                    return self.netMain(tenInput)
                # end

                if self.netShortcut is None:
                    return self.netMain(tenInput) + tenInput

                elif self.netShortcut is not None:
                    return self.netMain(tenInput) + self.netShortcut(tenInput)

                # end
            # end
        # end

        class Downsample(torch.nn.Module):
            def __init__(self, intChannels):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
                    torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        class Upsample(torch.nn.Module):
            def __init__(self, intChannels):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
                    torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        class Encode(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=32, init=0.25),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=32, init=0.25)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=64, init=0.25),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=64, init=0.25)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=96, init=0.25),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, bias=False),
                    torch.nn.PReLU(num_parameters=96, init=0.25)
                )
            # end

            def forward(self, tenInput):
                tenOutput = []

                tenOutput.append(self.netOne(tenInput))
                tenOutput.append(self.netTwo(tenOutput[-1]))
                tenOutput.append(self.netThr(tenOutput[-1]))

                return [torch.cat([tenInput, tenOutput[0]], 1)] + tenOutput[1:]
            # end
        # end

        class Softmetric(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netInput = torch.nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False)
                self.netError = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False)

                for intRow, intFeatures in [(0, 16), (1, 32), (2, 64), (3, 96)]:
                    self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True))
                # end

                for intCol in [0]:
                    self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([16, 32, 32]))
                    self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([32, 64, 64]))
                    self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([64, 96, 96]))
                # end

                for intCol in [1]:
                    self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([96, 64, 64]))
                    self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([64, 32, 32]))
                    self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([32, 16, 16]))
                # end

                self.netOutput = Basic('conv-relu-conv', [16, 16, 1], True)
            # end

            def forward(self, tenEncone, tenEnctwo, tenFlow):
                tenColumn = [None, None, None, None]

                tenColumn[0] = torch.cat([self.netInput(tenEncone[0][:, 0:3, :, :]), self.netError(torch.nn.functional.l1_loss(input=tenEncone[0], target=backwarp(tenEnctwo[0], tenFlow), reduction='none').mean([1], True))], 1)
                tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0])
                tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
                tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2])

                intColumn = 1
                for intRow in range(len(tenColumn) -1, -1, -1):
                    tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
                    if intRow != len(tenColumn) - 1:
                        tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

                        if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                        if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)

                        tenColumn[intRow] = tenColumn[intRow] + tenUp
                    # end
                # end

                return self.netOutput(tenColumn[0])
            # end
        # end

        class Warp(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = Basic('conv-relu-conv', [3 + 3 + 32 + 32 + 1 + 1, 32, 32], True)
                self.netTwo = Basic('conv-relu-conv', [0 + 0 + 64 + 64 + 1 + 1, 64, 64], True)
                self.netThr = Basic('conv-relu-conv', [0 + 0 + 96 + 96 + 1 + 1, 96, 96], True)
            # end

            def forward(self, tenEncone, tenEnctwo, tenMetricone, tenMetrictwo, tenForward, tenBackward):
                tenOutput = []

                for intLevel in range(3):
                    if intLevel != 0:
                        tenMetricone = torch.nn.functional.interpolate(input=tenMetricone, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False)
                        tenMetrictwo = torch.nn.functional.interpolate(input=tenMetrictwo, size=(tenEnctwo[intLevel].shape[2], tenEnctwo[intLevel].shape[3]), mode='bilinear', align_corners=False)

                        tenForward = torch.nn.functional.interpolate(input=tenForward, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False) * (float(tenEncone[intLevel].shape[3]) / float(tenForward.shape[3]))
                        tenBackward = torch.nn.functional.interpolate(input=tenBackward, size=(tenEnctwo[intLevel].shape[2], tenEnctwo[intLevel].shape[3]), mode='bilinear', align_corners=False) * (float(tenEnctwo[intLevel].shape[3]) / float(tenBackward.shape[3]))
                    # end

                    tenOutput.append([self.netOne, self.netTwo, self.netThr][intLevel](torch.cat([
                        softsplat.softsplat(tenIn=torch.cat([tenEncone[intLevel], tenMetricone], 1), tenFlow=tenForward, tenMetric=tenMetricone.neg().clip(-20.0, 20.0), strMode='soft'),
                        softsplat.softsplat(tenIn=torch.cat([tenEnctwo[intLevel], tenMetrictwo], 1), tenFlow=tenBackward, tenMetric=tenMetrictwo.neg().clip(-20.0, 20.0), strMode='soft')
                    ], 1)))
                # end

                return tenOutput
            # end
        # end

        self.netEncode = Encode()

        self.netSoftmetric = Softmetric()

        self.netWarp = Warp()

        for intRow, intFeatures in [(0, 32), (1, 64), (2, 96)]:
            self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True))
            self.add_module(str(intRow) + 'x1' + ' - ' + str(intRow) + 'x2', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True))
            self.add_module(str(intRow) + 'x2' + ' - ' + str(intRow) + 'x3', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True))
            self.add_module(str(intRow) + 'x3' + ' - ' + str(intRow) + 'x4', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True))
            self.add_module(str(intRow) + 'x4' + ' - ' + str(intRow) + 'x5', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True))
        # end

        for intCol in [0, 1, 2]:
            self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([32, 64, 64]))
            self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([64, 96, 96]))
        # end

        for intCol in [3, 4, 5]:
            self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([96, 64, 64]))
            self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([64, 32, 32]))
        # end

        self.netOutput = Basic('conv-relu-conv', [32, 32, 3], True)
    # end

    def forward(self, tenOne, tenTwo, tenForward, tenBackward, fltTime):
        tenEncone = self.netEncode(tenOne)
        tenEnctwo = self.netEncode(tenTwo)

        tenMetricone = self.netSoftmetric(tenEncone, tenEnctwo, tenForward) * 2.0 * fltTime
        tenMetrictwo = self.netSoftmetric(tenEnctwo, tenEncone, tenBackward) * 2.0 * (1.0 - fltTime)

        tenForward = tenForward * fltTime
        tenBackward = tenBackward * (1.0 - fltTime)

        tenWarp = self.netWarp(tenEncone, tenEnctwo, tenMetricone, tenMetrictwo, tenForward, tenBackward)

        tenColumn = [None, None, None]

        tenColumn[0] = tenWarp[0]
        tenColumn[1] = tenWarp[1] + self._modules['0x0 - 1x0'](tenColumn[0])
        tenColumn[2] = tenWarp[2] + self._modules['1x0 - 2x0'](tenColumn[1])

        intColumn = 1
        for intRow in range(len(tenColumn)):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != 0:
                tenColumn[intRow] = tenColumn[intRow] + self._modules[str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow - 1])
            # end
        # end

        intColumn = 2
        for intRow in range(len(tenColumn)):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != 0:
                tenColumn[intRow] = tenColumn[intRow] + self._modules[str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow - 1])
            # end
        # end

        intColumn = 3
        for intRow in range(len(tenColumn) -1, -1, -1):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != len(tenColumn) - 1:
                tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

                if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)

                tenColumn[intRow] = tenColumn[intRow] + tenUp
            # end
        # end

        intColumn = 4
        for intRow in range(len(tenColumn) -1, -1, -1):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != len(tenColumn) - 1:
                tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

                if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)

                tenColumn[intRow] = tenColumn[intRow] + tenUp
            # end
        # end

        intColumn = 5
        for intRow in range(len(tenColumn) -1, -1, -1):
            tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
            if intRow != len(tenColumn) - 1:
                tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

                if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, -1, 0, 0], mode='constant', value=0.0)

                tenColumn[intRow] = tenColumn[intRow] + tenUp
            # end
        # end

        return self.netOutput(tenColumn[0])
    # end
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netFlow = Flow()

        self.netSynthesis = Synthesis()

        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/softsplat/network-' + arguments_strModel + '.pytorch', file_name='softsplat-' + arguments_strModel).items()})
    # end

    def forward(self, tenOne, tenTwo, fltTimes):
        with torch.set_grad_enabled(False):
            tenStats = [tenOne, tenTwo]
            tenMean = sum([tenIn.mean([1, 2, 3], True) for tenIn in tenStats]) / len(tenStats)
            tenStd = (sum([tenIn.std([1, 2, 3], False, True).square() + (tenMean - tenIn.mean([1, 2, 3], True)).square() for tenIn in tenStats]) / len(tenStats)).sqrt()
            tenOne = ((tenOne - tenMean) / (tenStd + 0.0000001)).detach()
            tenTwo = ((tenTwo - tenMean) / (tenStd + 0.0000001)).detach()
        # end

        objFlow = self.netFlow(tenOne, tenTwo)

        tenImages = [self.netSynthesis(tenOne, tenTwo, objFlow['tenForward'], objFlow['tenBackward'], fltTime) for fltTime in fltTimes]

        return [(tenImage * tenStd) + tenMean for tenImage in tenImages]
    # end
# end

netNetwork = None

##########################################################

def estimate(tenOne, tenTwo, fltTimes):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    # end

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    intPadr = (2 - (intWidth % 2)) % 2
    intPadb = (2 - (intHeight % 2)) % 2

    tenPreprocessedOne = torch.nn.functional.pad(input=tenPreprocessedOne, pad=[0, intPadr, 0, intPadb], mode='replicate')
    tenPreprocessedTwo = torch.nn.functional.pad(input=tenPreprocessedTwo, pad=[0, intPadr, 0, intPadb], mode='replicate')

    return [tenImage[0, :, :intHeight, :intWidth].cpu() for tenImage in netNetwork(tenPreprocessedOne, tenPreprocessedTwo, fltTimes)]
# end
##########################################################
import logging

logger = logging.getLogger(__name__)

raft = None

class Raft:
    def __init__(self):
        from torchvision.models.optical_flow import (Raft_Large_Weights,
                                                     raft_large)

        weights = Raft_Large_Weights.DEFAULT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = raft_large(weights=weights, progress=False).to(self.device)
        self.model = model.eval()

    def __call__(self,img1,img2):
        with torch.no_grad():
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            i1 = torch.vstack([img1,img2])
            i2 = torch.vstack([img2,img1])
            list_of_flows = self.model(i1, i2)

        predicted_flows = list_of_flows[-1]
        return { 'tenForward' : predicted_flows[0].unsqueeze(dim=0) , 'tenBackward' : predicted_flows[1].unsqueeze(dim=0) }

img_count = 0
def debug_save_img(img, comment, inc=False):
    return
    global img_count
    from torchvision.utils import save_image

    save_image(img, f"debug0/{img_count:04d}_{comment}.png")

    if inc:
        img_count += 1


class Network2(torch.nn.Module):
    def __init__(self, model_file_path):
        super().__init__()

        self.netFlow = Flow()

        self.netSynthesis = Synthesis()

        d = torch.load(model_file_path)

        d = {strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in d.items()}

        self.load_state_dict(d)
    # end

    def forward(self, tenOne, tenTwo, guideFrameList):
        global raft

        do_composite = True
        use_raft = True

        if use_raft:
            if raft is None:
                raft = Raft()


        with torch.set_grad_enabled(False):
            tenStats = [tenOne, tenTwo]
            tenMean = sum([tenIn.mean([1, 2, 3], True) for tenIn in tenStats]) / len(tenStats)
            tenStd = (sum([tenIn.std([1, 2, 3], False, True).square() + (tenMean - tenIn.mean([1, 2, 3], True)).square() for tenIn in tenStats]) / len(tenStats)).sqrt()
            tenOne = ((tenOne - tenMean) / (tenStd + 0.0000001)).detach()
            tenTwo = ((tenTwo - tenMean) / (tenStd + 0.0000001)).detach()

            gtenStats = guideFrameList
            gtenMean = sum([tenIn.mean([1, 2, 3], True) for tenIn in gtenStats]) / len(gtenStats)
            gtenStd = (sum([tenIn.std([1, 2, 3], False, True).square() + (gtenMean - tenIn.mean([1, 2, 3], True)).square() for tenIn in gtenStats]) / len(gtenStats)).sqrt()
            guideFrameList = [((g - gtenMean) / (gtenStd + 0.0000001)).detach() for g in guideFrameList]

        # end

        tenImages =[]
        l = len(guideFrameList)
        i = 1
        g1 = guideFrameList.pop(0)

        if use_raft:
            styleFlow = raft(tenOne, tenTwo)
        else:
            styleFlow = self.netFlow(tenOne, tenTwo)

        def composite1(fA, fB, nA, nB):
            # 1,2,768,512
            A = fA[:,0,:,:]
            B = fA[:,1,:,:]
            Z = nA

            UA = A / Z
            UB = B / Z

            A2 = fB[:,0,:,:]
            B2 = fB[:,1,:,:]
            Z2 = nB
            fB[:,0,:,:] = Z2 * UA
            fB[:,1,:,:] = Z2 * UB
            return fB

        def mask_dilate(ten, kernel_size=3):
            ten = ten.to(torch.float32)
            k=torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32).cuda()
            ten = torch.nn.functional.conv2d(ten, k, padding=(kernel_size//2, kernel_size// 2))
            result = torch.clamp(ten, 0, 1)
            return result.to(torch.bool)

        def composite2(fA, fB, nA, nB):
            Z = nA
            Z2 = nB

            mean2 = torch.mean(Z2)
            max2 = torch.max(Z2)
            mask2 = (Z2 > (mean2+max2)/2)
            debug_save_img(mask2.to(torch.float), "mask2_0")
            mask2 = mask_dilate(mask2, 9)
            debug_save_img(mask2.to(torch.float), "mask2_1")
            mask2 = ~mask2

            debug_save_img(mask2.to(torch.float), "mask2")

            mean1 = torch.mean(Z)
            max1 = torch.max(Z)
            mask1 = (Z > (mean1+max1)/2)

            debug_save_img(mask1.to(torch.float), "mask1")

            mask = mask1 & mask2
            mask = mask.squeeze()

            debug_save_img(mask.to(torch.float), "cmask", True)

            fB[:,:,mask] = fA[:,:,mask]

            return fB


        def composite(fA, fB):
            A = fA[:,0,:,:]
            B = fA[:,1,:,:]
            Z = (A*A + B*B)**0.5
            A2 = fB[:,0,:,:]
            B2 = fB[:,1,:,:]
            Z2 = (A2*A2 + B2*B2)**0.5

            fB = composite1(fA, fB, Z, Z2)
            fB = composite2(fA, fB, Z, Z2)
            return fB

        for g2 in guideFrameList:
            if use_raft:
                objFlow = raft(g1, g2)
            else:
                objFlow = self.netFlow(g1, g2)


            objFlow['tenForward'] = objFlow['tenForward'] * (l/i)
            objFlow['tenBackward'] = objFlow['tenBackward'] * (l/i)

            if do_composite:
                objFlow['tenForward'] = composite(objFlow['tenForward'], styleFlow['tenForward'])
                objFlow['tenBackward'] = composite(objFlow['tenBackward'], styleFlow['tenBackward'])

            img = self.netSynthesis(tenOne, tenTwo, objFlow['tenForward'], objFlow['tenBackward'], i/l)
            tenImages.append(img)
            i += 1

        return [(tenImage * tenStd) + tenMean for tenImage in tenImages]


# end

netNetwork = None

##########################################################

def estimate2(img1: PIL.Image, img2:PIL.Image, guideFrames, model_file_path):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network2(model_file_path).cuda().eval()
    # end

    def forTensor(im):
        return torch.FloatTensor(numpy.ascontiguousarray(numpy.array(im)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

    tenOne = forTensor(img1)
    tenTwo = forTensor(img2)

    tenGuideFrames=[]
    for g in guideFrames:
        tenGuideFrames.append(forTensor(g))

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)
    tenGuideFrames = [ ten.cuda().view(1, 3, intHeight, intWidth) for ten in tenGuideFrames]

    intPadr = (2 - (intWidth % 2)) % 2
    intPadb = (2 - (intHeight % 2)) % 2

    tenPreprocessedOne = torch.nn.functional.pad(input=tenPreprocessedOne, pad=[0, intPadr, 0, intPadb], mode='replicate')
    tenPreprocessedTwo = torch.nn.functional.pad(input=tenPreprocessedTwo, pad=[0, intPadr, 0, intPadb], mode='replicate')
    tenGuideFrames = [ torch.nn.functional.pad(input=ten, pad=[0, intPadr, 0, intPadb], mode='replicate') for ten in tenGuideFrames]

    result = [tenImage[0, :, :intHeight, :intWidth].cpu() for tenImage in netNetwork(tenPreprocessedOne, tenPreprocessedTwo, tenGuideFrames)]
    result = [ PIL.Image.fromarray((r.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)) for r in result]

    return result
# end

##########################################################
'''
if __name__ == '__main__':
    if arguments_strOut.split('.')[-1] in ['bmp', 'jpg', 'jpeg', 'png']:
        tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
        tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

        tenOutput = estimate(tenOne, tenTwo, [0.5])[0]

        PIL.Image.fromarray((tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(arguments_strOut)

    elif arguments_strOut.split('.')[-1] in ['avi', 'mp4', 'webm', 'wmv']:
        import moviepy
        import moviepy.editor
        import moviepy.video.io.ffmpeg_writer

        objVideoreader = moviepy.editor.VideoFileClip(filename=arguments_strVideo)
        objVideoreader2 = moviepy.editor.VideoFileClip(filename=arguments_strVideo2)

        from moviepy.video.fx.resize import resize
        objVideoreader2 = resize(objVideoreader2, (objVideoreader.w, objVideoreader.h))

        intWidth = objVideoreader.w
        intHeight = objVideoreader.h

        tenFrames = [None, None, None, None]

        with moviepy.video.io.ffmpeg_writer.FFMPEG_VideoWriter(filename=arguments_strOut, size=(intWidth, intHeight), fps=objVideoreader.fps) as objVideowriter:
            for npyFrame in objVideoreader.iter_frames():
                tenFrames[3] = torch.FloatTensor(numpy.ascontiguousarray(npyFrame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

                if tenFrames[0] is not None:
                    tenFrames[1:3] = estimate(tenFrames[0], tenFrames[3], [0.333, 0.666])

                    objVideowriter.write_frame((tenFrames[0].clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8))
                    objVideowriter.write_frame((tenFrames[1].clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8))
                    objVideowriter.write_frame((tenFrames[2].clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8))
#                    objVideowriter.write_frame((tenFrames[3].clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8))
                # end

                tenFrames[0] = torch.FloatTensor(numpy.ascontiguousarray(npyFrame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
            # end
        # end

    # end
# end
'''
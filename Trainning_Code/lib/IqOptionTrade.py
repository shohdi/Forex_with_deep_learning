import pyautogui
import time


class IqOptionTrade:
    def __init__(self):
        None
    
    def _doTrade(self,direction):
        width = pyautogui.size().width
        height = pyautogui.size().height
        x=int((1202/1600) * width)
        y=int((212/900) * height)
        pyautogui.moveTo(x=x,y=y)
        pyautogui.click()
        pyautogui.click()
        x=int((1530/1600) * width)
        y=int((135/900) * height)
        pyautogui.moveTo(x=x,y=y)
        pyautogui.click()
        time.sleep(1/10)
        x=int((1243/1600) * width)
        y=int((355/900) * height)
        #y=int((307/900) * height)
        pyautogui.moveTo(x=x,y=y)
        pyautogui.click()
        #time.sleep(3/10)
        if direction == 1:
            x=int((1538/1600) * width)
            y=int((427/900) * height)
            pyautogui.moveTo(x=x,y=y)
            pyautogui.click()
        if direction == 2:
            x=int((1538/1600) * width)
            y=int((548/900) * height)
            pyautogui.moveTo(x=x,y=y)
            pyautogui.click()
    

    def doUpTrade(self):
        self._doTrade(1)
    
    def doDownTrade(self):
        self._doTrade(2)



if __name__ == "__main__":
    time.sleep(5)
    testObj = IqOptionTrade()
    testObj.doUpTrade()
    time.sleep(5)
    testObj.doDownTrade()
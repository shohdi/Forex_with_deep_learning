import pyautogui
import time


class IqOptionTrade:
    def __init__(self):
        None
    
    def _doTrade(self,direction):
        width = pyautogui.size().width
        height = pyautogui.size().height
        x=int((1536/1600) * width)
        y=int((224/900) * height)
        pyautogui.moveTo(x=x,y=y)
        pyautogui.click()
        time.sleep(5/10)
        x=int((1235/1600) * width)
        y=int((441/900) * height)
        pyautogui.moveTo(x=x,y=y)
        pyautogui.click()
        time.sleep(3/10)
        if direction == 1:
            x=int((1537/1600) * width)
            y=int((504/900) * height)
            pyautogui.moveTo(x=x,y=y)
            pyautogui.click()
        if direction == 2:
            x=int((1537/1600) * width)
            y=int((639/900) * height)
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
"""Test file to verify BOT_PAT workflow integration."""
import os,sys,json
from typing import Dict,List,Optional

def poorly_formatted_function(x,y,z):
    """This function has bad formatting that black will fix."""
    result=x+y+z
    if result>10:
        return True
    else:
        return False

class BadlyFormattedClass:
    def __init__(self,name,value):
        self.name=name
        self.value=value

    def get_info(self)->Dict:
        return {"name":self.name,"value":self.value}

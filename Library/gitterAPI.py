"""
The purpose of this module is to provide a reusable interface to crawl
gitter for comments
"""

from gitterpy.client import GitterClient
import code
import time

# setup the tokens
token = None
with open('/home/a2sachs/Documents/Library/gitterToken.txt','r') as f:
    token = f.read().strip()
gitter = GitterClient(token)
tokens = None
with open('/home/a2sachs/Documents/Library/gitterTokens.txt','r') as f:
    lst = f.readlines()
    tokens = [x.strip() for x in lst]
tokenI = 0

gitters = []
for t in tokens:
    gitters.append(GitterClient(t))

def getGitter():
    """
    (getGitter) Returns the gitter client with the next rolling token

    getGitter: None -> GitterClient(<next token>)
    ex.
    getGitter() -> GitterClient('1dfbhwk...')
    """
    global tokenI
    if tokenI < len(tokens):
        ans = gitters[tokenI]
        tokenI += 1
    else:
        ans = gitters[0]
        tokenI = 0
    return ans

def getRoomNameUri(repo):
    """
    (getRoomName repo) This function returns the unique name string of the room associated with repo
    (getRoomName str) -> str,str
    Ex:
    getRoomName('Oni') -> 'Oni/Lobby',...
    """
    print('getRoomName of',repo)
    prefix,suffix = repo.split('/')
    stringsToTry = [repo,'{0}/Lobby'.format(suffix),'{0}/Lobby'.format(suffix[0].upper()+suffix[1:])]
    name = None
    for s in stringsToTry:
        ret = None
        try:
            ret = getGitter().rooms.grab_room(s)
        except:
            print('sleeping...')
            time.sleep(10)
            ret = getGitter().rooms.grab_room(s)
        print('ret',ret)
        if ret == {'error':'Too Many Requests'}:
            print('sleeping...')
            time.sleep(10)
            return getRoomNameUri(repo)
        if ret not in [{'error':'Not Found'},{'error':'Moved Permanently'}]:
            print('second ret',ret)
            return ret['name'],s
    return None

def getMessages(roomName):
    """
    (getMessages roomName) -> [{'name':...,'message':...,'datetime':...},...] This function returns a list of messages up to 'till' time
    getMessages: str, str -> (listof (dictof str:str))
    Ex.
    getMessages('Oni',...) -> [{'name': 'bryphe', 'message': "i'm re-running to see", 'datetime': '2018-02-26T23:17:46.151Z'}, {'name': 'bryphe', 'message': "I would imagine it's a fluke, since your PR build passed", 'datetime': '2018-02-26T23:17:53.831Z'}, {'name': 'Akin909', 'message': 'Will definitely join another night, its near midnight where I am :laughing: ', 'datetime': '2018...
    """
    allMessages = getGitter().messages.list(roomName)
    ans = []
    for msg in allMessages:
        if 'fromUser' in msg: # if 'fromUser' exists then this is an actual message (as opposed to just a notification of something from github
            entry = {
                    'name':msg['fromUser']['username'],
                    'message':msg['text'],
                    'datetime':msg['sent']
                    }
            ans.append(entry)
            # they seem to be in ascending order (dateimewise, oldest at 0, newest at -1)
    return ans

def joinRooms(lst):
    """
    (joinRooms lst) This function joins the rooms whose names are listed in lst (uris) for all of the
    rolling tokens
    joinRooms: (listof str) -> None, Effect: joins all the listed rooms
    ex.
    joinRooms(['hadley/devtools','pritunl/pritunl']) -> None
    """
    for i,name in enumerate(lst):
        for client in gitters:
            client.rooms.join(name)
        print('all joined',name, i+1, 'of',len(lst))

def leaveAllRooms():
    """
    (leaveAllRooms lst) This function leaves all the rooms possible all of the
    rolling tokens
    joinRooms: None -> None, Effect: leaves all rooms
    ex.
    leaveAllRooms() -> None
    """
    for client in gitters:
        lst = [x['name'] for x in client.rooms.rooms_list]
        for name in lst:
            client.rooms.leave(name)

if __name__ == '__main__':
    g = getGitter()
    code.interact(local=locals())

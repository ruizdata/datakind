# desktop_app

#imports
import time
import datetime
from tkinter import *
from tkcalendar import Calendar
from tktimepicker import SpinTimePickerModern
import os
import pandas as pd
import numpy as np
import subprocess as sub
from tkinter import filedialog

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.document_loaders.json_loader import JSONLoader
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.llms import AzureOpenAI
from langchain.llms import OpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
#from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import langchain
from typing import Dict, Any, Tuple
from langchain.memory.utils import get_prompt_input_key
import shutil
import os
import sys
import json
from googletrans import Translator
from dotenv import load_dotenv



# A little mod to enable using memory *and* getting docs. See: https://github.com/langchain-ai/langchain/issues/2256#issuecomment-1665188576
def _get_input_output(
    self, inputs: Dict[str, Any], outputs: Dict[str, str]
) -> Tuple[str, str]:
    if self.input_key is None:
        prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
    else:
        prompt_input_key = self.input_key
    if self.output_key is None:
        output_key = list(outputs.keys())[0]
    else:
        output_key = self.output_key
    return inputs[prompt_input_key], outputs[output_key]


# Annoyingly, and unlike OpenAI direct, some API settings differ depending on service used
def set_azure_env(type="embedding"):
    os.environ["OPENAI_API_TYPE"] = "azure"
    fields = ["OPENAI_API_BASE", "OPENAI_API_KEY"]
    if type == "embedding":
        os.environ["OPENAI_API_VERSION"]=os.getenv("EMBEDDING_API_VERSION")
    else:
        os.environ["OPENAI_API_VERSION"]=os.getenv("CHAT_API_VERSION")
    for k in fields:
        os.environ[k] = os.getenv(k)
      

# tkinter 'after()' function
def after():
    print("after")
    root.after(60000,after)     


#### MAIN APP CLASS ########
class App(Frame):
    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.vecs_dir = "/vecs"
        self.data_file = ""
        self.filetypes = [".pdf"]
        self.temperature = 0.0
        
        if(os.path.isdir(self.vecs_dir) == False):
            os.mkdir(self.vecs_dir)

        # Widget Variables
        self.server_port = IntVar(value=8500)
        self.server_address = StringVar(value='127.0.0.1')        
        
        
        # Functions

        # ensure input port is valid and available
        def validate_port(port: int) -> bool:
            if(1024 <= port <= 65535):
                return True
            else:
                return False
                
        # ensure input address is valid
        def validate_address(address: str) -> bool:
            #if(address.isValidCIDR()):
                #return True
            #else:
                #return False
            return        

        def launch_server(command):         
            result = sub.Popen(command, shell=True, text=True)
            print(result)         
            #result = sub.run(command, shell=True, capture_output=True, text=True)  
            #print(result[0])
            #print(result[1])
            #print(result[2])    

            

        def import_data():
            print()
            import_files = filedialog.askopenfilenames(initialdir=os.path.curdir)

            # parse filenames and exclude those not in self.filetypes
            
           
            for file in import_files:
                print(file.split(".")[1])
                if(file.split(".")[1] == "pdf"):

                    # then perform the loading/chunking/vec creation, etcetera if extracting dictionary
                    pdf = file
                    print(f"Reading the PDF {pdf}...")
                    loader = PyPDFLoader(pdf)
                    documents = loader.load()
                    print("Chunking text into documents ...")
                    text_splitter = CharacterTextSplitter(separator="\n",chunk_size=2000,chunk_overlap=500,length_function=len)
                    docs = text_splitter.split_documents(documents)
                    print("Done")
                    print("number of docs in file is: ", len(docs))

                    # The rest of Matt's Embedding/VectorDB code here
                    # 
                    #  Else, Raj's code here if extracting tables

            # OR,... for import could just keep it simple and store/display file location of the data and verify extension compatibility
            # then have other functions allowing user to select the desired operation


          
            

            return import_files  

        # Widgets
        main_menu = Menu(root, background='grey', type='menubar',title='main menu',)
        main_menu.configure(tearoff=False, font=("Arial",8))
        main_menu.pack(fill='x')


        ######## LEFT FRAME ############
        left_frame = Frame(root, background='grey', width=400)
        left_frame.pack(side='left',fill='both', padx=5, pady=10, expand=True)


        ## IMPORT NEW DATASET ##    
        import_frame = Frame(left_frame, background='white', width=400)
        import_frame.configure(border=1, borderwidth=1, highlightcolor="Black", highlightbackground="Black",
                              highlightthickness=2,)
        import_frame.pack_propagate(False)
        import_frame.pack(anchor='w', fill='both', expand=True)


        import_label = Label(import_frame, text="IMPORT NEW DATASET", background='light grey',
                              foreground='black', font=("Arial",10))
        import_label.pack(fill='x')   


        import_btn = Button(import_label, background='orange', foreground='black', default='normal', state='normal',
                             width=10, height=1, text="IMPORT", command=import_data)
        import_btn.pack(side='right')  


        ## IMPORTED DATASETS ##
        imported_frame = Frame(left_frame, background='white', width=400)
        imported_frame.configure(border=1, borderwidth=1, highlightcolor="Black", highlightbackground="Black",
                              highlightthickness=2)
        imported_frame.pack_propagate(False)
        imported_frame.pack(anchor='w', fill='both', expand=True)


        imported_label = Label(imported_frame, text="IMPORTED DATASETS", background='light grey',
                              foreground='black', font=("Arial",10))
        imported_label.pack(fill='x')


    
        
        ######## RIGHT FRAME ############
        right_frame = Frame(root, background='grey',width=400)
        right_frame.pack(side='right', fill='both', padx=5, pady=10, expand=True)


        ## SERVER LAUNCH ##
        server_frame = Frame(right_frame,background='white',width=400)
        server_frame.configure(highlightcolor="Black", highlightbackground="Black",
                             highlightthickness=2)
        server_frame.pack(fill='both', expand=True)
        server_frame.pack_propagate(False)


        server_lbl = Label(server_frame, text="SERVER", background='light grey',
                                foreground='black', font=("Arial",10))
        
        server_lbl.pack(fill='x')


        self.app_list = [r"workstream1\desktop_app_demo\Pdf_app.py", r"workstream1\desktop_app_demo\app.py"]
        self.bat_list = [r"workstream1\desktop_app_demo\run_Pdf_app.bat", r"workstream1\desktop_app_demo\run_app.py.bat"]

        app1_frame = Frame(server_frame, background='white', )
        app1_frame.pack(anchor='w', pady=5)

        app1_lbl = Label(app1_frame, text=self.app_list[0], background='white', foreground='black',
                            justify='left', anchor='w', font=("Arial",10), state='normal')
        app1_lbl.grid(column=0,row=0)
        
        app1_btn = Button(app1_frame, background='light blue', foreground='white', command=lambda: launch_server(self.bat_list[0]), default='normal', text='Launch')
        app1_btn.grid(column=1, row=0)

        app2_frame = Frame(server_frame, background='white', )
        app2_frame.pack(anchor='w', pady=5)

        app2_lbl = Label(app2_frame, text=self.app_list[1], background='white', foreground='black',
                            justify='left', anchor='w', font=("Arial",10), state='normal')
        app2_lbl.grid(column=0,row=1)
        
        app2_btn = Button(app2_frame, background='light blue', foreground='white', command=lambda: launch_server(self.bat_list[1]), default='normal', text='Launch')
        app2_btn.grid(column=1, row=1)

        ## SETTINGS ##
        settings_frame = Frame(right_frame, background='white', width=400)
        settings_frame.configure(highlightcolor="Black", highlightbackground="Black",
                              highlightthickness=2)
        settings_frame.pack(fill='both', expand=True)
        settings_frame.pack_propagate(False)


        settings_lbl = Label(settings_frame,text="SETTINGS", background='light grey', foreground='black',
                              font=("Arial",10))
        settings_lbl.pack(fill='x')

        port_frame = Frame(settings_frame, background='white', )
        port_frame.pack(anchor='w', pady=5)


        port_lbl = Label(port_frame,text="Server Port:", background='white', foreground='black',
                            justify='left', anchor='w', font=("Arial",12))
        port_lbl.grid(column=0,row=0)


        port_entry = Entry(port_frame)
        port_entry.configure(background='white', foreground='black',font=("Arial",10), validatecommand=validate_port,
                            justify='center', state='normal', highlightbackground="Black",
                              highlightcolor="Black", highlightthickness=1, width=19, textvariable=self.server_port)
        port_entry.grid(column=1,row=0,padx=20)        

        addr_frame = Frame(settings_frame, background='white', )
        addr_frame.pack(anchor='w', pady=5)


        addr_lbl = Label(addr_frame,text="Server Address:", background='white', foreground='black',
                            justify='left', anchor='w', font=("Arial",12))
        addr_lbl.grid(column=0,row=0)


        addr_entry = Entry(addr_frame)
        addr_entry.configure(background='white', foreground='black',font=("Arial",10), validatecommand=validate_address,
                            justify='center', state='normal', highlightbackground="Black",
                              highlightcolor="Black", highlightthickness=1, width=19, textvariable=self.server_address)
        addr_entry.grid(column=1,row=0,padx=20)    

''' SETTINGS FOR STREAMLIT --> COULD HAVE THIS SECTION PROVIDE STREAMLIT CONFIGURATION

User sets config options in gui --> options are written or server launched with Subprocess
  
[global]

# By default, Streamlit checks if the Python watchdog module is available
# and, if not, prints a warning asking for you to install it. The watchdog
# module is not required, but highly recommended. It improves Streamlit's
# ability to detect changes to files in your filesystem.

# If you'd like to turn off this warning, set this to True.

# Default: false
# disableWatchdogWarning = false

# By default, Streamlit displays a warning when a user sets both a widget
# default value in the function defining the widget and a widget value via
# the widget's key in `st.session_state`.

# If you'd like to turn off this warning, set this to True.

# Default: false
# disableWidgetStateDuplicationWarning = false

# If True, will show a warning when you run a Streamlit-enabled script
# via "python my_script.py".

# Default: true
# showWarningOnDirectExecution = true

# DataFrame serialization.

# Acceptable values:
# - 'legacy': Serialize DataFrames using Streamlit's custom format. Slow
# but battle-tested.
# - 'arrow': Serialize DataFrames using Apache Arrow. Much faster and versatile.

# Default: "arrow"
# dataFrameSerialization = "arrow"


[logger]

# Level of logging: 'error', 'warning', 'info', or 'debug'.

# Default: 'info'
# level = "info"

# String format for logging messages. If logger.datetimeFormat is set,
# logger messages will default to `%(asctime)s.%(msecs)03d %(message)s`. See
# [Python's documentation](https://docs.python.org/2.6/library/logging.html#formatter-objects)
# for available attributes.

# Default: "%(asctime)s %(message)s"
# messageFormat = "%(asctime)s %(message)s"


[client]

# Whether to enable st.cache. This does not affect st.cache_data or
# st.cache_resource.

# Default: true
# caching = true

# If false, makes your Streamlit script not draw to a
# Streamlit app.

# Default: true
# displayEnabled = true

# Controls whether uncaught app exceptions and deprecation warnings
# are displayed in the browser. By default, this is set to True and
# Streamlit displays app exceptions and associated tracebacks, and
# deprecation warnings, in the browser.

# If set to False, deprecation warnings and full exception messages
# will print to the console only. Exceptions will still display in the
# browser with a generic error message. For now, the exception type and
# traceback show in the browser also, but they will be removed in the
# future.

# Default: true
# showErrorDetails = true

# Change the visibility of items in the toolbar, options menu,
# and settings dialog (top right of the app).

# Allowed values:
# * "auto" : Show the developer options if the app is accessed through
# localhost or through Streamlit Community Cloud as a developer.
# Hide them otherwise.
# * "developer" : Show the developer options.
# * "viewer" : Hide the developer options.
# * "minimal" : Show only options set externally (e.g. through
# Streamlit Community Cloud) or through st.set_page_config.
# If there are no options left, hide the menu.

# Default: "auto"
# toolbarMode = "auto"


[runner]

# Allows you to type a variable or string by itself in a single line of
# Python code to write it to the app.

# Default: true
# magicEnabled = true

# Install a Python tracer to allow you to stop or pause your script at
# any point and introspect it. As a side-effect, this slows down your
# script's execution.

# Default: false
# installTracer = false

# Sets the MPLBACKEND environment variable to Agg inside Streamlit to
# prevent Python crashing.

# Default: true
# fixMatplotlib = true

# Run the Python Garbage Collector after each script execution. This
# can help avoid excess memory use in Streamlit apps, but could
# introduce delay in rerunning the app script for high-memory-use
# applications.

# Default: true
# postScriptGC = true

# Handle script rerun requests immediately, rather than waiting for script
# execution to reach a yield point. This makes Streamlit much more
# responsive to user interaction, but it can lead to race conditions in
# apps that mutate session_state data outside of explicit session_state
# assignment statements.

# Default: true
# fastReruns = true

# Raise an exception after adding unserializable data to Session State.
# Some execution environments may require serializing all data in Session
# State, so it may be useful to detect incompatibility during development,
# or when the execution environment will stop supporting it in the future.

# Default: false
# enforceSerializableSessionState = false


[server]

# List of folders that should not be watched for changes. This
# impacts both "Run on Save" and @st.cache.

# Relative paths will be taken as relative to the current working directory.

# Example: ['/home/user1/env', 'relative/path/to/folder']

# Default: []
# folderWatchBlacklist = []

# Change the type of file watcher used by Streamlit, or turn it off
# completely.

# Allowed values:
# * "auto" : Streamlit will attempt to use the watchdog module, and
# falls back to polling if watchdog is not available.
# * "watchdog" : Force Streamlit to use the watchdog module.
# * "poll" : Force Streamlit to always use polling.
# * "none" : Streamlit will not watch files.

# Default: "auto"
# fileWatcherType = "auto"

# Symmetric key used to produce signed cookies. If deploying on multiple replicas, this should
# be set to the same value across all replicas to ensure they all share the same secret.

# Default: randomly generated secret key.
# cookieSecret = "6f12e688af79d3adb0b37f34222a0d1c18088b653f9082f9e34c0d140cd38e24"

# If false, will attempt to open a browser window on start.

# Default: false unless (1) we are on a Linux box where DISPLAY is unset, or
# (2) we are running in the Streamlit Atom plugin.
# headless = false

# Automatically rerun script when the file is modified on disk.

# Default: false
# runOnSave = false

# The address where the server will listen for client and browser
# connections. Use this if you want to bind the server to a specific address.
# If set, the server will only be accessible from this address, and not from
# any aliases (like localhost).

# Default: (unset)
# address =

# The port where the server will listen for browser connections.

# Default: 8501
# port = 8501

# The base path for the URL where Streamlit should be served from.

# Default: ""
# baseUrlPath = ""

# Enables support for Cross-Origin Resource Sharing (CORS) protection, for added security.

# Due to conflicts between CORS and XSRF, if `server.enableXsrfProtection` is on and
# `server.enableCORS` is off at the same time, we will prioritize `server.enableXsrfProtection`.

# Default: true
# enableCORS = true

# Enables support for Cross-Site Request Forgery (XSRF) protection, for added security.

# Due to conflicts between CORS and XSRF, if `server.enableXsrfProtection` is on and
# `server.enableCORS` is off at the same time, we will prioritize `server.enableXsrfProtection`.

# Default: true
# enableXsrfProtection = true

# Max size, in megabytes, for files uploaded with the file_uploader.

# Default: 200
# maxUploadSize = 200

# Max size, in megabytes, of messages that can be sent via the WebSocket connection.

# Default: 200
# maxMessageSize = 200

# Enables support for websocket compression.

# Default: false
# enableWebsocketCompression = false

# Enable serving files from a `static` directory in the running app's directory.

# Default: false
# enableStaticServing = false

# Server certificate file for connecting via HTTPS.
# Must be set at the same time as "server.sslKeyFile".

# ['DO NOT USE THIS OPTION IN A PRODUCTION ENVIRONMENT. It has not gone through security audits or performance tests. For the production environment, we recommend performing SSL termination by the load balancer or the reverse proxy.']
# sslCertFile =

# Cryptographic key file for connecting via HTTPS.
# Must be set at the same time as "server.sslCertFile".

# ['DO NOT USE THIS OPTION IN A PRODUCTION ENVIRONMENT. It has not gone through security audits or performance tests. For the production environment, we recommend performing SSL termination by the load balancer or the reverse proxy.']
# sslKeyFile =


[browser]

# Internet address where users should point their browsers in order to
# connect to the app. Can be IP address or DNS name and path.

# This is used to:
# - Set the correct URL for CORS and XSRF protection purposes.
# - Show the URL on the terminal
# - Open the browser

# Default: "localhost"
# serverAddress = "localhost"

# Whether to send usage statistics to Streamlit.

# Default: true
# gatherUsageStats = true

# Port where users should point their browsers in order to connect to the
# app.

# This is used to:
# - Set the correct URL for CORS and XSRF protection purposes.
# - Show the URL on the terminal
# - Open the browser

# Default: whatever value is set in server.port.
# serverPort = 8501


[mapbox]

# Configure Streamlit to use a custom Mapbox
# token for elements like st.pydeck_chart and st.map.
# To get a token for yourself, create an account at
# https://mapbox.com. It's free (for moderate usage levels)!

# Default: ""
# token = ""


[deprecation]

# Set to false to disable the deprecation warning for the file uploader encoding.

# Default: true
# showfileUploaderEncoding = true

# Set to false to disable the deprecation warning for using the global pyplot instance.

# Default: true
# showPyplotGlobalUse = true


[theme]

# The preset Streamlit theme that your custom theme inherits from.
# One of "light" or "dark".
# base =

# Primary accent color for interactive elements.
# primaryColor =

# Background color for the main content area.
# backgroundColor =

# Background color used for the sidebar and most interactive widgets.
# secondaryBackgroundColor =

# Color used for almost all text.
# textColor =

# Font family for all text in the app, except code blocks. One of "sans serif",
# "serif", or "monospace".
# font =
'''



      
############### RUN MAIN PROGRAM #######################
if __name__ == "__main__":

    # "Annoyingly, and unlike OpenAI direct, some API settings differ depending on service used"
    langchain.memory.chat_memory.BaseChatMemory._get_input_output = _get_input_output

    # load environment
    load_dotenv('workstream1\desktop_app_demo\.env.example')

    # Create Root Window
    root = Tk(className=' IMF Household Survey Data Importer')
    root.configure(background="grey",highlightbackground="Black",highlightcolor="Black")
    root.geometry("800x700")
    root.minsize(800,700)
    root.anchor('center')
    
    cur_datetime = datetime.datetime.now()


    # other variables
    app = App(root)
    app.pack(side="top", fill="both", expand=True)

    # sync 'after' function to beginning of the minute
    cur_datetime_micro = datetime.datetime.now().microsecond
    cur_seconds = datetime.datetime.now().second
    wait_micros = 1000000-cur_datetime_micro
    wait_seconds = 60 - cur_seconds
    root.after(int(wait_micros/1000)+(1000*wait_seconds), after)
    root.mainloop()
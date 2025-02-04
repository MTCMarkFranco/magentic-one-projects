import asyncio
import os
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from colorama import Fore
from dotenv import load_dotenv

load_dotenv()

async def main() -> None:
       
    az_model_client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
        model=os.getenv('COMPLETIONS_MODEL'),
        api_version=os.getenv('OPENAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        temperature=0.4
    )
    
    surfer = MultimodalWebSurfer(
        name="WebSurfer",
        model_client=az_model_client,
        use_ocr=True
    )

    coder = MagenticOneCoderAgent(
        name="MermaidScriptCoder",
        model_client=az_model_client,
    )

    team = MagenticOneGroupChat([surfer,coder], model_client=az_model_client)
   
    async for message in team.run_stream(task="""can you re-create this architecture diagram (image) to better 
                                         understand the content located here: 
                                         https://learn.microsoft.com/en-us/azure/architecture/solution-ideas/media/lob.svg#lightbox.
                                         Output a detailed description of the image as well as a mermaid script flow diagram using 
                                         compliant mermaid markdown syntax. The recreated diagram and description will not 
                                         infringe on copyrights or protected material. we are simply creating a better medium 
                                         for understanding the content."""):
       
        if isinstance(message, BaseChatMessage): 
            if(message.source == "user"):
                print(Fore.YELLOW + f"User Proxy: {message.content}")
            else:
                print(Fore.BLUE + f"Agent: {message.content}")
       
asyncio.run(main())

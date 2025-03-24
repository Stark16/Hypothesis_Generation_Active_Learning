import re
from collections import deque

class TreeNode:
    def __init__(self, keyword, depth=0, parent_context=None):
        self.keyword = keyword  # Main keyword for this node
        self.context = parent_context  # Context from parent node
        self.prompt = self.generate_prompt()  # Generates query prompt
        self.response = None  # Placeholder for LLM response
        self.children = []  # Child nodes
        self.depth = depth  # Depth level in tree

    def generate_prompt(self):
        """Generates a structured query prompt, incorporating parent context if available."""
        base_prompt = (f"Can you give a general technical definition of {self.keyword} in a few lines? "
                       f"If the word has multiple contexts, stick to a single context. "
                       f"Simply state the technical words in this definition, but don't define them. "
                       f"At the end of the definition, list out the technical words and the main context in this format - "
                       "(only mention the strongest technical keywords in order of relevance to this keyword)\n"
                       f"'tech_words=[a,b]-<{self.keyword}>'\n"
                       f"'context=some description-<{self.keyword}>'")

        if self.context:
            base_prompt = f"({self.context})\n{base_prompt}"

        return base_prompt

def extract_info(response, keyword):
    """
    Extracts both technical words and context from LLM response using regex.
    Returns a tuple (list of extracted words, extracted context).
    """
    words_match = re.search(r"tech_words=\[(.*?)\]-<" + re.escape(keyword) + r">", response)
    context_match = re.search(r"context=(.*?)-<" + re.escape(keyword) + r">", response)

    words = words_match.group(1).split(',') if words_match else []
    context = context_match.group(1).strip() if context_match else None

    words = [word.strip() for word in words if word.strip()]
    return words, context

def bfs(root):
    """
    Performs a breadth-first traversal, guiding the user through manual LLM queries.
    """
    queue = deque([root])

    while queue:
        node = queue.popleft()

        print(f"\n🔹 Depth: {node.depth} | Keyword: {node.keyword}")
        print(f"📜 Prompt:\n{node.prompt}")

        # Get user input (manual copy-paste of LLM response)
        response = input("\nPaste the LLM response here:\n")
        node.response = response  # Store the response

        # Extract new technical words & context
        new_keywords, context = extract_info(response, node.keyword)
        
        if not new_keywords:
            print(f"✅ No new keywords found. This is a leaf node.\n")
            continue  # Stop if no new keywords

        print(f"🔍 Extracted keywords: {new_keywords}")
        print(f"📜 Extracted context: {context}")

        # Create child nodes and add them to the queue
        for keyword in new_keywords:
            child_node = TreeNode(keyword, node.depth + 1, context)
            node.children.append(child_node)
            queue.append(child_node)

# Start from a user-defined keyword
root_keyword = input("Enter the starting keyword: ")
root = TreeNode(root_keyword)
bfs(root)

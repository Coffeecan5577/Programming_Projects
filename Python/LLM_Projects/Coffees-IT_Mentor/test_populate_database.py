from populate_database import main  # Assuming 'main' is a function in your original script

import unittest

class TestLoadDocuments(unittest.TestCase):
    def test_load_documents(self):
        # Set up some sample data (in this case, an empty list)
        data_path = 'sample_data.txt'
        documents = []

        # Call the load_documents function with our sample data
        loaded_documents = main.load_documents(data_path)

        # Verify that the loaded documents are not empty
        self.assertGreater(len(loaded_documents), 0)


'''
class TestSplitDocuments(unittest.TestCase):
    def test_split_documents(self):
    # Set up some sample data (in this case, a short text)
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        documents = [Document(text=text)]

    # Call the split_documents function with our sample data
    chunks = main.split_documents(documents)

    # Verify that the chunked documents have the correct length
    for chunk in chunks:
        self.assertEqual(len(chunk), 800)

'''

if __name__ == "__main__":
    unittest.main()
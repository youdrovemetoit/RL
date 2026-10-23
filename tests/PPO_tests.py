import unittest

class Test(unittest.TestCase):
    
    def test(self):
        s = 1 + 1
        self.assertEqual(2,2,"Maths is broken")

if __name__ == "__main__":
    unittest.main()

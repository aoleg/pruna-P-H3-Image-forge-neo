"""The CUDA error Forge reports as out of memory gets its traceback printed once."""

import contextlib
import io
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None or not hasattr(torch, "AcceleratorError"), "needs torch with AcceleratorError")
class DisguisedErrorTests(unittest.TestCase):
    def make(self, cls, message):
        try:
            return cls(message)
        except TypeError:
            # some torch error types take no plain message
            error = cls.__new__(cls)
            Exception.__init__(error, message)
            return error

    def test_only_a_non_oom_accelerator_error_counts(self):
        from forge_h3.native.cuda_errors import is_disguised_cuda_error
        illegal = self.make(torch.AcceleratorError, "CUDA error: an illegal memory access was encountered")
        self.assertTrue(is_disguised_cuda_error(illegal))
        self.assertFalse(is_disguised_cuda_error(self.make(torch.AcceleratorError, "CUDA error: out of memory")))
        self.assertFalse(is_disguised_cuda_error(self.make(torch.OutOfMemoryError, "CUDA out of memory")))
        self.assertFalse(is_disguised_cuda_error(RuntimeError("an illegal memory access")))

    def test_the_wrapper_keeps_forges_answer_and_prints_once(self):
        from forge_h3.native.cuda_errors import NOTE, report_disguised
        calls = []

        def forge_is_oom(error):
            calls.append(error)
            return isinstance(error, (torch.AcceleratorError, torch.OutOfMemoryError))

        wrapped = report_disguised(forge_is_oom)
        self.assertIs(wrapped.__wrapped__, forge_is_oom)
        illegal = self.make(torch.AcceleratorError, "CUDA error: an illegal memory access was encountered")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertTrue(wrapped(illegal))
            self.assertTrue(wrapped(illegal))
            self.assertFalse(wrapped(ValueError("x")))
            self.assertTrue(wrapped(self.make(torch.OutOfMemoryError, "CUDA out of memory")))
        self.assertEqual(out.getvalue().count(NOTE), 1)
        self.assertIn("illegal memory access", err.getvalue())
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()

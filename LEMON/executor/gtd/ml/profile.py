import json
import tensorflow as tf
from tensorflow.python.client import timeline


class ProfiledSession(tf.Session):
    def __init__(self, *args, **kwargs):
        super(ProfiledSession, self).__init__(*args, **kwargs)

    def run(self, fetches, feed_dict=None):
        """like Session.run, but return a Timeline object in Chrome trace format (JSON).

        Save the json to a file, go to chrome://tracing, and open the file.

        Args:
            fetches
            feed_dict

        Returns:
            dict: a JSON dict
        """
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        super(ProfiledSession, self).run(fetches, feed_dict, options=options, run_metadata=run_metadata)

        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        return json.loads(ctf)
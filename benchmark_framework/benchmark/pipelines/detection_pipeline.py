from benchmark.pipelines.preprocess import YOLOPreprocessor
from benchmark.pipelines.postprocess import YOLOPostprocessor


class DetectionPipeline:
    def __init__(self, adapter, profiler, config, visualizer=None):
        self.adapter = adapter
        self.profiler = profiler
        self.config = config
        self.preprocessor = YOLOPreprocessor(config)
        self.postprocessor = YOLOPostprocessor(config)
        self.visualizer = visualizer
        self._raw_output_examples = []

    def run_single(self, dataset_item):
        raw_output = None

        with self.profiler.profile_stage("full_pipeline"):
            with self.profiler.profile_stage("image_read"):
                image = dataset_item.image

            backend_name = self.adapter.get_backend_name()

            if backend_name == "tensorrt":
                with self.profiler.profile_stage("preprocess"):
                    prep = None

                with self.profiler.profile_stage("inference"):
                    raw_output = self.adapter.infer(image)

                with self.profiler.profile_stage("postprocess"):
                    predictions = self.postprocessor(
                        dataset_item,
                        raw_output,
                        self.adapter,
                        None,
                    )

            elif backend_name == "rknn":
                with self.profiler.profile_stage("preprocess"):
                    prep = self.preprocessor(image)

                with self.profiler.profile_stage("inference"):
                    rknn_input = prep.tensor.detach().cpu().numpy()
                    raw_output = self.adapter.infer(rknn_input)

                with self.profiler.profile_stage("postprocess"):
                    predictions = self.postprocessor(
                        dataset_item,
                        raw_output,
                        self.adapter,
                        prep.meta,
                    )

            else:
                with self.profiler.profile_stage("preprocess"):
                    prep = self.preprocessor(image)

                with self.profiler.profile_stage("inference"):
                    raw_output = self.adapter.infer(prep.tensor)

                with self.profiler.profile_stage("postprocess"):
                    predictions = self.postprocessor(
                        dataset_item,
                        raw_output,
                        self.adapter,
                        prep.meta,
                    )

            self._run_visualization(dataset_item, predictions)

        self.profiler.record_value("num_predictions", len(predictions))

        if raw_output is not None and len(self._raw_output_examples) < 3:
            self._raw_output_examples.append(
                self.postprocessor.summarize_raw_output(raw_output)
            )

        return predictions

    def _run_visualization(self, dataset_item, predictions):
        vis_cfg = self.config.get("visualization", {})
        enabled = vis_cfg.get("enabled", False)

        if not enabled or self.visualizer is None:
            return

        with self.profiler.profile_stage("visualization"):
            rendered = self.visualizer.render(dataset_item, predictions)

            if vis_cfg.get("show_window", False):
                with self.profiler.profile_stage("imshow"):
                    self.visualizer.imshow(rendered)

                with self.profiler.profile_stage("waitkey"):
                    self.visualizer.wait_key(vis_cfg.get("wait_key_ms", 1))

    def get_debug_info(self):
        return {
            "raw_output_examples": self._raw_output_examples
        }
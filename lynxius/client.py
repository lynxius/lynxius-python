"""Main module."""

import os
import time
from urllib.parse import urljoin

import httpx
from httpx import HTTPStatusError, RequestError

from lynxius.datasets.types import Dataset, DatasetDetails, DatasetEntry
from lynxius.evals.evaluator import Evaluator


class LynxiusClient:
    LYNXIUS_API_VERSION = "v1"

    _client: httpx.Client

    # client options
    api_key: str

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        run_local: bool | None = False,
    ) -> None:
        """Construct a new synchronous lynxius client instance.

        This automatically infers the following arguments from their corresponding
        environment variables if they are not provided:
        - `api_key` from `LYNXIUS_API_KEY`
        """
        if api_key is None:
            api_key = os.environ.get("LYNXIUS_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to \
                    the client or by setting the LYNXIUS_API_KEY environment variable"
            )
        self.api_key = api_key

        if base_url is None:
            base_url = os.environ.get("LYNXIUS_BASE_URL")
        if base_url is None:
            base_url = "https://platform.lynxius.ai"

        base_url = urljoin(base_url, "api/")
        base_url = urljoin(base_url, self.LYNXIUS_API_VERSION)
        # Now, base_url looks similar to this: https://platform.lynxius.ai/api/v1"

        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Determines if evals are run locally or remotely
        self.run_local = run_local

        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            follow_redirects=True,
        )

    def evaluate(self, eval: Evaluator) -> str | None:
        """
        Initiates a batched evaluation job. Returns an eval run ID.
        """

        # Local evaluation
        if self.run_local:
            eval.evaluate_local()

        response = self._client.post(
            eval.get_url(run_local=self.run_local),
            json=eval.get_request_body(run_local=self.run_local),
        )

        if response.status_code == httpx.codes.CREATED:
            return response.json()["uuid"]
        else:
            print("Error:", response.status_code, response.text)
            return None

    def get_eval_run(self, eval_run_uuid: str) -> Evaluator | None:
        """
        Returns the details of an Eval Run.
        Retries for up to 30 seconds before failing.
        """
        timeout = 30
        backoff_factor = 1
        start_time = time.time()

        attempt = 0

        while time.time() - start_time < timeout:
            attempt += 1
            try:
                response = self._client.get(
                    f"/projects/evals/{eval_run_uuid}/"
                ).raise_for_status()
                body = response.json()
                if body.get("status") == "SUCCESS":
                    return body
                else:
                    print(
                        f"Attempt {attempt} received status {body.get('status')}. Retrying..."
                    )
            except (HTTPStatusError, RequestError) as exc:
                print(f"Attempt {attempt} failed: {exc}. Retrying...")

            sleep_time = backoff_factor * (2 ** (attempt - 1))
            time.sleep(sleep_time)

        print(f"All attempts within {timeout} seconds failed.")
        return None

    def get_dataset_details(self, dataset_id: str) -> DatasetDetails:
        response = self._client.get(
            f"/datasets/{dataset_id}/entries/"
        ).raise_for_status()
        body = response.json()

        dataset_details = DatasetDetails()
        dataset_details.dataset = Dataset(
            body["dataset"]["uuid"],
            body["dataset"]["date_created"],
            body["dataset"]["organization_uuid"],
            body["dataset"]["organization_name"],
        )

        dataset_details.entries = []
        for entry in body["entries"]:
            dataset_entry = DatasetEntry(
                entry["uuid"],
                entry["dataset_uuid"],
                entry["query"],
                entry["output"],
                entry["reference"],
                entry["score"],
                entry["comments"],
                entry["date_created"],
                entry["date_modified"],
            )

            dataset_details.entries.append(dataset_entry)

        return dataset_details

"""Main module."""

import os
from urllib.parse import urljoin

import httpx

from lynxius.evals.evaluator import Evaluator
from lynxius.datasets.types import Dataset, DatasetDetails, DatasetEntry


class LynxiusClient:
    LYNXIUS_API_VERSION = "v1"

    _client: httpx.Client

    # client options
    api_key: str

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
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
            # TODO: substitute production URL here
            base_url = "https://api.lynxius.ai/"

        base_url = urljoin(base_url, "api/")
        base_url = urljoin(base_url, self.LYNXIUS_API_VERSION)
        # Now, base_url looks similar to this: https://lynxius.ai/api/v1"

        headers = {"Authorization": f"Bearer {self.api_key}"}

        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            follow_redirects=True,
        )

    def evaluate(self, eval: Evaluator) -> str | None:
        """
        Initiates a batched evaluation job. Returns an eval run ID.
        """

        response = self._client.post(eval.get_url(), json=eval.get_request_body())

        if response.status_code == httpx.codes.CREATED:
            return response.json()["uuid"]
        else:
            print("Error:", response.status_code, response.text)
            return None

    def get_dataset_details(self, dataset_id: str) -> DatasetDetails:
        response = self._client.get(f"/datasets/{dataset_id}/entries/")
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

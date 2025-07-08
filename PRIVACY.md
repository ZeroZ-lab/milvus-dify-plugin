# Privacy Policy for Milvus Plugin

## Data Collection and Usage

The Milvus Plugin for Dify acts as a connector between the Dify platform and your Milvus vector database instance. This plugin:

1. **Does not collect or store** any user data on its own servers.
2. **Processes data** only as necessary to execute the requested operations between Dify and your Milvus database.
3. **Transmits data** directly between the Dify platform and your specified Milvus instance without intermediate storage.

## Credentials Handling

When you configure the plugin with your Milvus credentials:

1. These credentials are **stored securely** within your Dify instance.
2. Credentials are **never transmitted** to the plugin developer or any third parties.
3. All authentication tokens are **used solely** for establishing connections to your specified Milvus instance.

## Data Processing

The plugin processes the following types of data:

1. **Vector data** that you explicitly provide for storage, retrieval, or search operations.
2. **Metadata** associated with your vectors as provided by you.
3. **Collection information** such as names, schemas, and statistics.

## Security Measures

To ensure the security of your data:

1. All communications between the plugin and your Milvus instance use **secure connections** as configured in your Milvus setup.
2. The plugin implements **proper error handling** to prevent unintended data exposure.
3. No operational logs containing your data are stored persistently by the plugin.

## Third-Party Services

This plugin connects only to:

1. Your specified **Milvus database instance** as configured by you.
2. No other third-party services are contacted during normal operation.

## Data Retention

The plugin does not retain any data beyond the immediate processing required to fulfill your requests. All data processing is transient and exists only for the duration of the specific operation.

## Updates to Privacy Policy

This privacy policy may be updated periodically. Any changes will be reflected in the plugin's documentation and release notes.

## Contact Information

If you have questions or concerns about this privacy policy or the data handling practices of this plugin, please contact the plugin developer through the GitHub repository.

---

Last Updated: July 2025